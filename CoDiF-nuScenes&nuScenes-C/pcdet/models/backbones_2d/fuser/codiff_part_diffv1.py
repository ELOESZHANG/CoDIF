import os
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision
import numpy as np
from tqdm import tqdm
import scipy.signal as signal


class Upsample(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.num_groups = num_groups

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")  # # 上采样
        x = self.conv(x)  # # 卷积 + GroupNorm
        return x  # 激活函数


class Downsample(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv(x)  # 卷积 + GroupNorm
        return x  # 激活函数


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        h = F.relu(F.group_norm(self.conv1(x), num_groups=32))  # 第一层卷积 + GroupNorm + 激活
        h = F.relu(F.group_norm(self.conv2(h), num_groups=32))
        return h + self.shortcut(x)  # 残差连接


def timestep_embedding(t, dim, max_period=10000):
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=dim // 2, dtype=torch.float32) / (dim // 2)).to(
        device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding


class UNetModel(nn.Module):
    def __init__(self, io_channels=128, model_channels=64, class_num=None, class_free=None):
        super().__init__()
        self.model_channels = model_channels

        self.down_block1 = nn.Conv2d(io_channels, model_channels, kernel_size=3, padding=1)
        self.down_block2 = Downsample(model_channels)
        self.down_block3 = nn.Conv2d(model_channels, model_channels * 2, kernel_size=3, padding=1)
        self.down_block4 = Downsample(model_channels * 2)

        self.middle_block = ResidualBlock(model_channels * 2, model_channels * 2)
        self.noise_embedding = nn.Linear(model_channels, model_channels * 2)
        # if class_free != None:
        #     self.class_emb = nn.Embedding(class_num, model_channels * 2)

        self.up_block1 = Upsample(model_channels * 2)
        self.up_block2 = nn.Conv2d(model_channels * 2, model_channels, kernel_size=3, padding=1)
        self.up_block3 = Upsample(model_channels)
        self.up_block4 = nn.Conv2d(model_channels, io_channels, kernel_size=3, padding=1)

    def forward(self, x, t, cond):
        # 输入
        x1 = F.relu(F.group_norm(self.down_block1(x), num_groups=32))
        x2 = F.relu(F.group_norm(self.down_block2(x1), num_groups=32))
        x3 = F.relu(F.group_norm(self.down_block3(x2), num_groups=32))
        x4 = F.relu(F.group_norm(self.down_block4(x3), num_groups=32))
        # 条件
        c1 = F.relu(F.group_norm(self.down_block1(cond), num_groups=32))
        c2 = F.relu(F.group_norm(self.down_block2(c1), num_groups=32))
        c3 = F.relu(F.group_norm(self.down_block3(c2), num_groups=32))
        c4 = F.relu(F.group_norm(self.down_block4(c3), num_groups=32))

        middle = self.middle_block(x4)
        noise_t = F.relu(self.noise_embedding(timestep_embedding(t, self.model_channels)))

        middle = middle + c4 + noise_t[:, :, None, None]

        x5 = F.relu(F.group_norm(self.up_block1(middle + x4), num_groups=32))
        x6 = F.relu(F.group_norm(self.up_block2(x5 + x3), num_groups=32))
        x7 = F.relu(F.group_norm(self.up_block3(x6 + x2), num_groups=32))
        out = self.up_block4(x7 + x1)

        return out


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = 0.0003 * scale  # 该值不可过小，去燥不充分
    beta_end = 0.03 * scale  # 该值不可过小，乱序条纹
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)




class PhysicalNoiseGenerator:
    """物理噪声生成器，修改为动态设备分配"""

    def generate_vibration_noise(self, shape, t=None, vibration_freq=10.0, vibration_amp=0.1, device=None):
        """
        生成振动噪声
        Args:
            shape: 噪声形状
            t: 时间步（可为 None）
            device: 设备，None 时自动推断
        """
        if device is None:
            if t is not None:
                device = t.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, channels, height, width = shape
        y_coords = torch.linspace(0, 1, height, device=device)
        x_coords = torch.linspace(0, 1, width, device=device)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        noise = torch.zeros(shape, device=device)
        for freq_mult in [1.0, 2.0, 3.0]:
            phase = torch.rand(1, device=device) * 2 * math.pi
            if t is not None:
                phase = phase + t.float().mean() * 0.01  # 使用均值避免 batch 维度问题
            angle = torch.rand(1, device=device) * 2 * math.pi
            direction = torch.cos(angle) * x_grid + torch.sin(angle) * y_grid
            vibration = vibration_amp * torch.sin(2 * math.pi * vibration_freq * freq_mult * direction + phase)
            noise += vibration.unsqueeze(0).unsqueeze(0).expand(shape)
        high_freq_noise = torch.randn(shape, device=device) * 0.05
        noise += high_freq_noise
        return noise

    def generate_calibration_error(self, shape, offset_scale=0.15, bias_type='gradient', device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, channels, height, width = shape
        if bias_type == 'uniform':
            offset = torch.randn(batch_size, channels, 1, 1, device=device) * offset_scale
            bias = offset.expand(shape)
        elif bias_type == 'gradient':
            y_coords = torch.linspace(-1, 1, height, device=device)
            x_coords = torch.linspace(-1, 1, width, device=device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            gradient_x = torch.randn(batch_size, channels, 1, 1, device=device) * offset_scale
            gradient_y = torch.randn(batch_size, channels, 1, 1, device=device) * offset_scale
            bias = gradient_x * x_grid.unsqueeze(0).unsqueeze(0) + gradient_y * y_grid.unsqueeze(0).unsqueeze(0)
        elif bias_type == 'patch':
            bias = torch.zeros(shape, device=device)
            patch_size = max(1, min(height, width) // 4)
            for b in range(batch_size):
                for c in range(channels):
                    num_patches = torch.randint(1, 4, (1,)).item()
                    for _ in range(num_patches):
                        patch_h = torch.randint(0, height - patch_size, (1,)).item()
                        patch_w = torch.randint(0, width - patch_size, (1,)).item()
                        patch_offset = torch.randn(1, device=device) * offset_scale
                        bias[b, c, patch_h:patch_h+patch_size, patch_w:patch_w+patch_size] += patch_offset
        else:
            raise ValueError(f"Unknown bias_type: {bias_type}")
        return bias

    def generate_temperature_noise(self, shape, temp_variation=0.05, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        batch_size, channels, height, width = shape
        center_y, center_x = height // 2, width // 2
        y_coords = torch.arange(height, device=device).float()
        x_coords = torch.arange(width, device=device).float()
        distance = torch.sqrt((y_coords[:, None] - center_y)**2 + (x_coords[None, :] - center_x)**2)
        max_dist = math.sqrt(center_y**2 + center_x**2)
        temperature = torch.exp(-distance**2 / (2 * (max_dist/3)**2))
        temp_noise = temperature.unsqueeze(0).unsqueeze(0) * torch.randn(batch_size, channels, 1, 1, device=device) * temp_variation
        return temp_noise.expand(shape)

    def generate_physical_noise(self, shape, t=None, noise_components=None, device=None):
        if device is None:
            if t is not None:
                device = t.device
            else:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if noise_components is None:
            noise_components = ['vibration', 'calibration', 'temperature']
        total_noise = torch.zeros(shape, device=device)
        weights = {'vibration': 0.4, 'calibration': 0.4, 'temperature': 0.2}
        if 'vibration' in noise_components:
            vibration_noise = self.generate_vibration_noise(shape, t, device=device)
            total_noise += weights['vibration'] * vibration_noise
        if 'calibration' in noise_components:
            bias_types = ['uniform', 'gradient', 'patch']
            bias_type = bias_types[torch.randint(0, len(bias_types), (1,)).item()]
            calibration_noise = self.generate_calibration_error(shape, bias_type=bias_type, device=device)
            total_noise += weights['calibration'] * calibration_noise
        if 'temperature' in noise_components:
            temperature_noise = self.generate_temperature_noise(shape, device=device)
            total_noise += weights['temperature'] * temperature_noise
        random_noise = torch.randn(shape, device=device) * 0.05
        total_noise += random_noise
        if total_noise.std() > 0:
            total_noise = (total_noise - total_noise.mean()) / total_noise.std()
        return total_noise



class GaussianDiffusion:
    def __init__(self, timesteps=1000, use_physical_noise=True):
        self.timesteps = timesteps
        self.betas = linear_beta_schedule(timesteps)
        self.betas_cumprod = torch.cumprod(self.betas, axis=0)

        self.alphas = 1. - self.betas  # 接近1的数
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1.0 - self.alphas_cumprod)

        # 物理噪声生成器
        self.use_physical_noise = use_physical_noise
        if use_physical_noise:
            self.noise_generator = PhysicalNoiseGenerator()

    def _extract(self, a: torch.FloatTensor, t: torch.LongTensor, x_shape):
        # get the param of given timestep t
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start: torch.FloatTensor, t: torch.LongTensor, noise=None):
        # 前向加噪过程：forward diffusion (using the nice property): q(x_t | x_0)
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        if noise is None:
            if self.use_physical_noise:
                # 使用物理噪声
                noise = self.noise_generator.generate_physical_noise(
                    x_start.shape,
                    t=t,
                    noise_components=['vibration', 'calibration', 'temperature']
                )
            else:
                # 使用随机高斯噪声
                noise = torch.randn_like(x_start)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def train_losses(self, model, x_start: torch.FloatTensor, t: torch.LongTensor, cond: torch.FloatTensor):
        if self.use_physical_noise:
            # 使用物理噪声
            noise = self.noise_generator.generate_physical_noise(
                x_start.shape,
                t=t,
                noise_components=['vibration', 'calibration', 'temperature'],
                device=x_start.device
            )
        else:
            # 使用随机高斯噪声
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start, t, noise=noise)  # x_t ~ q(x_t | x_0)
        predicted_noise = model(x_noisy, t, cond=cond)  # predict noise from noisy image
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    # DDPM Inference/Reverse
    def q_posterior_mean_variance(self, x_start: torch.FloatTensor, x_t: torch.FloatTensor, t: torch.LongTensor):
        # Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)
        posterior_mean = (self._extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + self._extract(
            self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = self._extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, c=None):
        # compute x_0 from x_t and pred noise: the reverse of `q_sample`, 估计值，包含部分残留噪声
        pre_x_0 = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - self._extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * model(x_t, t, c)  # pred_noise = model(x_t, t)
        pre_x_0 = torch.clamp(pre_x_0, min=-1., max=1.)  # clip_denoised
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_mean_variance(pre_x_0, x_t,
                                                                                                t)  ## compute predicted mean and variance of p(x_{t-1} | x_t), predict noise using model
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, model, x_t: torch.FloatTensor, t: torch.LongTensor, c=None):
        # denoise_step: sample x_{t-1} from x_t and pred_noise, predict mean and variance
        model_mean, posterior_variance, model_log_variance = self.p_mean_variance(model, x_t, t, c)

        if self.use_physical_noise:
            # 使用物理噪声
            noise = self.noise_generator.generate_physical_noise(
                x_t.shape,
                t=t,
                noise_components=['vibration', 'calibration', 'temperature'],
                device=x_t.device
            )
        else:
            noise = torch.randn_like(x_t)

        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1))))  # no noise when t == 0
        pred_img = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise  # compute x_{t-1}
        return pred_img

    def sample(self, model: nn.Module, image_size, batch_size=8, channels=3, n_class=None, device=None):
        if device is None:
            device = next(model.parameters()).device
        shape = (batch_size, channels, image_size[0], image_size[1])  # denoise: reverse diffusion

        if self.use_physical_noise:
            # 使用物理噪声作为初始噪声
            img = self.noise_generator.generate_physical_noise(
                shape,
                t=torch.full((batch_size,), self.timesteps - 1, device=device),
                noise_components=['vibration', 'calibration', 'temperature'],
                device=device,
            )
        else:
            img = torch.randn(shape,
                              device=device)  # start from pure noise (for each example in the batch), x_T ~ N(0, 1)

        imgs = []
        if n_class != None:
            cur_y = torch.randint(0, n_class, (batch_size,)).to(device)  # 随机标签
        else:
            cur_y = None
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, c=cur_y)
            imgs.append(img)
        return imgs

    # DDIM Inference/Reverse
    def ddim_sample(self, model, image_size, cond, ddim_timesteps=100, batch_size=64, channels=3, n_class=1, device='cuda:0'):

        shape = (batch_size, channels, image_size[0], image_size[1])

        if self.use_physical_noise:
            # 使用物理噪声作为初始噪声
            x_T = self.noise_generator.generate_physical_noise(
                shape,
                t=torch.full((batch_size,), self.timesteps - 1, device=device),
                noise_components=['vibration', 'calibration', 'temperature'],
                device=device
            )
        else:
            x_T = torch.randn(shape, device=device)  # start from pure noise

        xs = [x_T]
        c = self.timesteps // ddim_timesteps  # make ddim timestep sequence
        ddim_timestep_seq = torch.tensor(
            list(range(0, self.timesteps, c))) + 1  # one from first scale to data during sampling
        ddim_timestep_prev_seq = torch.cat((torch.tensor([0]), ddim_timestep_seq[:-1]))  # previous sequence

        # for i in tqdm(reversed(range(0, ddim_timesteps)), desc='ddpm sampling loop time step', total=ddim_timesteps):
        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t,
                                            x_T.shape)  # 1. get current and previous alpha_cumprod
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, next_t, x_T.shape)
            # print('t:',t.shape)
            # print('xs[-1]:', xs[-1].shape)
            # print('cond:', cond.shape)
            pred_noise = model(xs[-1], t, cond)  # 2. predict noise using model, 模型预测噪声
            pred_x0 = (xs[-1] - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)  # 3. get the predicted x_0, 预测 x_0
            pred_dir_xt = torch.sqrt(
                1 - alpha_cumprod_t_prev) * pred_noise  # 5. compute "direction pointing to x_t" of formula (12)
            x_t_pre = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt  # 6. compute x_{t-1} of formula (12)
            xs.append(x_t_pre)
            # omit 4. compute variance: "sigma_t(η)" -> see formula (16) / σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        return xs

    def feature_ddim_sample(self, model, original_features, cond, ddim_timesteps=6, n_class=1, start_step=0,
                            end_step=100,
                            device='cuda:0'):
        """
        特征图去噪的DDIM采样
        Args:
            model: 去噪模型
            original_features: 原始特征图 [batch_size, channels, height, width]
            ddim_timesteps: 采样步数
            start_step: 起始步数
            end_step: 结束步数
            device: 设备
        """

        batch_size, channels, height, width = original_features.shape

        # 计算步长间隔
        c = (end_step - start_step) // (ddim_timesteps - 1)
        ddim_timestep_seq = torch.tensor(list(range(start_step, end_step + c, c)))
        ddim_timestep_prev_seq = torch.cat((torch.tensor([0]), ddim_timestep_seq[:-1]))

        # 方案A：假设 original_features 是 x_0，先加噪到最大时间步
        t_max = torch.full((batch_size,), end_step, device=device, dtype=torch.long)

        if self.use_physical_noise:
            # 使用物理噪声
            noise = self.noise_generator.generate_physical_noise(
                original_features.shape,
                t=t_max,
                noise_components=['vibration', 'calibration', 'temperature'],
                device=device
            )
        else:
            noise = torch.randn_like(original_features)

        x_start = original_features  # 假设 original_features 是相对干净的
        x_t = self.q_sample(x_start, t_max, noise=noise)

        # 反向去噪过程：从噪声特征逐步恢复
        xs = [x_t]  # 从最大噪声的特征开始

        for i in reversed(range(0, ddim_timesteps)):
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            next_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)

            alpha_cumprod_t = self._extract(self.alphas_cumprod, t,
                                            original_features.shape)  # 1. get current and previous alpha_cumprod
            alpha_cumprod_t_prev = self._extract(self.alphas_cumprod, next_t, original_features.shape)

            pred_noise = model(xs[-1], t, cond)  # 2. predict noise using model, 模型预测噪声
            pred_x0 = (xs[-1] - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            pred_x0 = torch.clamp(pred_x0, min=-1., max=1.)  # 3. get the predicted x_0, 预测 x_0
            pred_dir_xt = torch.sqrt(
                1 - alpha_cumprod_t_prev) * pred_noise  # 5. compute "direction pointing to x_t" of formula (12)
            x_t_pre = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + pred_dir_xt  # 6. compute x_{t-1} of formula (12)
            xs.append(x_t_pre)
        return xs


class CondiDifFusion(nn.Module):
    def __init__(self, args):
        super(CondiDifFusion, self).__init__()
        if self.training:
            self.channels = 64  # 128
        else:
            self.channels = 64
        # if self.training:
        #     self.image_size = [128, 216]
        self.n_class = 1

        if 'train_steps' in args.keys():
            # print('aaaaaaaaaaaaaaaaa')
            # print(args)
            self.timesteps = args['train_steps']
            self.ddim_steps = args['sample_stepss']
        else:
            self.timesteps = 400
            self.ddim_steps = 5  # 5

        self.unet = UNetModel(io_channels=self.channels, model_channels=64, class_num=self.n_class, class_free=True)
        self.gaussian_diffusion = GaussianDiffusion(timesteps=self.timesteps, use_physical_noise=True)

    def sample(self, images, condition):

        batch_size = images.shape[0]
        image_size = images.shape[2:]

        generated_images = self.gaussian_diffusion.ddim_sample(self.unet, image_size, cond=condition,
                                                               ddim_timesteps=self.ddim_steps, n_class=self.n_class,
                                                               batch_size=batch_size, channels=self.channels,
                                                               device=images.device)
        generated_images = (generated_images[-1] + 1) / 2

        return generated_images

    def denoise_sample(self, images, condition):

        # print('ddim_steps:',self.ddim_steps)
        generated_images = self.gaussian_diffusion.feature_ddim_sample(self.unet, images, cond=condition,
                                                                       ddim_timesteps=self.ddim_steps,
                                                                       n_class=self.n_class,
                                                                       start_step=0, end_step=self.timesteps - 1,
                                                                       device=images.device)  # end_step=5
        generated_images = (generated_images[-1] + 1) / 2

        return generated_images

    def forward(self, images, condition):

        batch_size = images.shape[0]
        t = torch.randint(0, self.timesteps, (batch_size,), device=images.device).long()
        loss = self.gaussian_diffusion.train_losses(self.unet, images, t, condition)
        # print('lossaaaabbb',loss.item())

        return loss


if __name__ == '__main__':
    config = {
        'train_steps': 400,  # 500
        'sample_stepss': 5
    }

    batch_size = 1
    image_size = [180, 180]  # cifar10  = 32 or fmnist 28
    timesteps = 400  # fmnist, mnist ddim_steps = 300/500, cifar=1000
    # in_channels = 16  # fmnist, mnist = 1, cifar = 3

    train = True  # False is inferences via pre-trained model
    class_free = True
    n_class = 1  # or None
    ddim_steps = 5

    # BEV: [1,256,200,176] ;ROI 1*128*128*256

    # Initialize the model and move to CUDA
    device = torch.device('cuda:0')

    model = UNetModel(io_channels=128, model_channels=64, class_num=1, class_free=True).to(device)
    gaussian_diffusion = GaussianDiffusion(timesteps=timesteps, use_physical_noise=True)
    from torchinfo import summary

    print(summary(model))

    model.train()

    # sample t uniformally for every example in the batch
    t = torch.randint(0, timesteps, (batch_size,), device=device).long()
    images = torch.randn(batch_size, 128, image_size[0], image_size[1]).to(device)
    condition = torch.randn(batch_size, 128, image_size[0], image_size[1]).to(device)
    loss = gaussian_diffusion.train_losses(model, images, t, cond=condition)
    print('loss', loss)

    with torch.no_grad():
        model.eval()
        generated_images = gaussian_diffusion.ddim_sample(model, image_size, cond=condition, ddim_timesteps=ddim_steps,
                                                          n_class=n_class,
                                                          batch_size=batch_size, channels=128)
        generated_images = (generated_images[-1] + 1) / 2
        print(generated_images.shape)

        denoised_features = gaussian_diffusion.feature_ddim_sample(model, images, cond=condition,
                                                                   ddim_timesteps=ddim_steps,
                                                                   n_class=n_class, start_step=0, end_step=5,
                                                                   # end_step=5,
                                                                   device=device)
        denoised_features = (denoised_features[-1] + 1) / 2
        print(denoised_features.shape)
