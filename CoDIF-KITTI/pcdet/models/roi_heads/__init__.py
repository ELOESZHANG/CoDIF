from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate
from .voxelrcnn_head import VoxelRCNNHead
from .sfd_head import SFDHead
from .mpcf_head import MPCFHead
from .mpcf_longsf_head import MPCFHead_longsf
from .codiff_head import CODIFFhead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'VoxelRCNNHead': VoxelRCNNHead,
    'SFDHead': SFDHead,
    'MPCFHead':MPCFHead,
    'MPCFHead_longsf':MPCFHead_longsf,
    'CODIFFhead':CODIFFhead,
}
