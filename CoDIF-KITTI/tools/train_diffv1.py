import os
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from pcdet.models.roi_heads.codiff_part_diffv1 import CondiDifFusion
from tqdm import tqdm
import argparse


class TrainConfig:
    CoDiff_CFG = {
        'train_steps': 400,
        'sample_stepss': 5
    }


def collate_fn(batch):
    # batch 是一个 list，长度为 batch_size（这里为1）
    return batch[0]  # 返回 (pseudo, valid)，形状不变


if __name__ == '__main__':
    # ---------- 参数解析 ----------
    parser = argparse.ArgumentParser(description='分布式扩散模型训练')
    parser.add_argument('--resume', type=str, default=None,
                        help='从指定检查点恢复训练，例如: --resume checkpoint_epoch10.pth')
    parser.add_argument('--data_dir', type=str,
                        default="/home/gaopan/GP_Third_Method/MPCF_Codiff/data/nuS_for_coDIFF",
                        help='数据目录路径')
    parser.add_argument('--num_epochs', type=int, default=500, help='总训练轮数')
    parser.add_argument('--batch_size', type=int, default=1, help='每个GPU的批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='模型保存目录')
    parser.add_argument('--save_every', type=int, default=5,
                       help='每多少个epoch保存一次历史checkpoint（默认：5）')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                       choices=['cosine', 'linear', 'none'],
                       help='学习率调整策略: cosine（余弦退火）, linear（线性下降）, none（固定LR）')
    parser.add_argument('--lr_min_ratio', type=float, default=0.01,
                       help='LR下降的最小比例（相对于初始LR），仅对cosine/linear生效')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪最大范数（默认：1.0，设为0则禁用）')
    parser.add_argument('--preload', action='store_true', default=False,
                        help='将数据预加载到内存，加速训练')
    parser.add_argument('--local-rank', type=int, default=0,
                        help='本地进程编号（由 torchrun 自动设置）')
    args = parser.parse_args()

    # ---------- 分布式初始化 ----------
    local_rank = args.local_rank
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # ---------- 配置 ----------
    cfg = TrainConfig()
    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.lr
    num_workers = args.num_workers
    pin_memory = True
    save_every = args.save_every  # 保存频率

    # 创建保存目录（仅rank 0）
    if local_rank == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    # ---------- 自定义 Dataset ----------
    class DiffusionDataset(Dataset):
        def __init__(self, data_dir, preload=False):
            self.preload = preload
            pairs = []
            for i in range(1, 20001):
                pseudo_path = os.path.join(data_dir, f"pseudo_{i:06d}.pt")
                valid_path = os.path.join(data_dir, f"valid_{i:06d}.pt")
                if os.path.exists(pseudo_path) and os.path.exists(valid_path):
                    pairs.append((pseudo_path, valid_path))
            self.pairs = pairs

            if preload:
                if local_rank == 0:
                    print(f"预加载 {len(self.pairs)} 对数据到内存...")
                self.all_pseudo = []
                self.all_valid = []
                for pseudo_path, valid_path in self.pairs:
                    self.all_pseudo.append(torch.load(pseudo_path, map_location='cpu'))
                    self.all_valid.append(torch.load(valid_path, map_location='cpu'))
                if local_rank == 0:
                    print("预加载完成")
            elif local_rank == 0:
                print(f"Found {len(self.pairs)} pairs in {data_dir}")

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            if self.preload:
                return self.all_pseudo[idx], self.all_valid[idx]
            pseudo_path, valid_path = self.pairs[idx]
            pseudo = torch.load(pseudo_path, map_location='cpu')
            valid = torch.load(valid_path, map_location='cpu')
            return pseudo, valid


    # ---------- 数据集 & 分布式采样器 ----------
    dataset = DiffusionDataset(data_dir, preload=args.preload)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True,
        seed=42
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )

    # ---------- 模型 & 优化器 ----------
    model = CondiDifFusion(cfg.CoDiff_CFG).to(device)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ---------- 学习率调度器（T_max/total_iters 在断点恢复时会修正） ----------
    lr_min = learning_rate * args.lr_min_ratio
    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=lr_min
        )
    elif args.lr_scheduler == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=args.lr_min_ratio,
            total_iters=num_epochs
        )
    else:
        scheduler = None

    if local_rank == 0:
        print(f"LR scheduler: {args.lr_scheduler} "
              f"(initial LR: {learning_rate}, min LR: {lr_min:.6f}, epochs: {num_epochs})")

    # ---------- 断点恢复 ----------
    start_epoch = 0
    loss_history = []
    if args.resume and os.path.exists(args.resume):
        if local_rank == 0:
            print(f"从检查点恢复: {args.resume}")

        # 加载检查点
        checkpoint = torch.load(args.resume, map_location=device)

        # 加载模型状态
        model.module.load_state_dict(checkpoint['model_state_dict'])

        # 加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 加载调度器状态，并调整 T_max 为剩余 epoch 数
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            loaded_epoch = checkpoint['epoch']
            remaining = max(1, num_epochs - loaded_epoch)
            if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=remaining, eta_min=lr_min
                )
            elif isinstance(scheduler, torch.optim.lr_scheduler.LinearLR):
                scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=1.0, end_factor=args.lr_min_ratio,
                    total_iters=remaining
                )
            if local_rank == 0:
                print(f"已重建设调度器（剩余 {remaining} epochs），当前LR: {scheduler.get_last_lr()[0]:.8f}")

        # 设置起始epoch
        start_epoch = checkpoint['epoch'] + 1

        # 加载损失历史（可选）
        if 'loss_history' in checkpoint:
            loss_history = checkpoint['loss_history']
            if local_rank == 0:
                print(f"已加载损失历史，最近损失: {loss_history[-1]:.6f}")

        if local_rank == 0:
            print(f"从第 {start_epoch} 轮继续训练")

    # ---------- 训练循环 ----------
    model.train()

    for epoch in range(start_epoch, num_epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据重排
        total_loss = 0.0
        num_samples = 0

        # 进度条：仅在 rank 0 显示
        pbar = tqdm(dataloader, disable=local_rank != 0,
                    desc=f"Epoch {epoch + 1}/{num_epochs}")

        for pseudo, valid in pbar:
            # pseudo/valid 形状: (batch, 64, 180, 180)
            pseudo = pseudo.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            loss = model(valid, pseudo) + model(pseudo, valid)

            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # 累加 loss（按样本数加权）
            batch_size_actual = pseudo.size(0)
            total_loss += loss.item() * batch_size_actual
            num_samples += batch_size_actual

            # 更新进度条的后缀（显示当前 batch loss 和 LR）
            if local_rank == 0:
                current_lr_bar = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss.item(), lr=f"{current_lr_bar:.8f}")

        # 同步所有进程的 loss 总和与样本总数
        loss_sum = torch.tensor([total_loss], device=device)
        sample_sum = torch.tensor([num_samples], device=device)
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(sample_sum, op=dist.ReduceOp.SUM)
        avg_loss = (loss_sum / sample_sum).item()
        loss_history.append(avg_loss)

        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = learning_rate

        if local_rank == 0:
            lr_info = f", LR: {current_lr:.8f}" if scheduler is not None else ""
            print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.6f}{lr_info}")

        # 保存检查点（仅在 rank 0，每 save_every 个 epoch）
        if local_rank == 0 and (epoch + 1) % save_every == 0:
            ckpt_dict = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_history,
                'config': cfg.CoDiff_CFG,
            }
            if scheduler is not None:
                ckpt_dict['scheduler_state_dict'] = scheduler.state_dict()
            latest_path = os.path.join(args.save_dir, "latest_checkpoint.pth")
            torch.save(ckpt_dict, latest_path)
            checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save(ckpt_dict, checkpoint_path)
            print(f"保存检查点到: {checkpoint_path}")

    # 清理分布式进程组
    dist.destroy_process_group()
