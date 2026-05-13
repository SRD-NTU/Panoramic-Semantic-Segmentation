import torch
from torch import nn
from tqdm import tqdm

from trimesh_utils import IcoSphereRef


class SphereNeighborhood(nn.Module):
    def __init__(self, rank: int, icosphere_ref: IcoSphereRef, win_size_coef: int):
        super().__init__()

        mapping = icosphere_ref.get_neighbor_mapping(rank=rank, num_hops=win_size_coef)
        num_nodes = len(mapping)
        num_keys = max(len(keys) for keys in mapping)

        idx = torch.arange(0, num_nodes).unsqueeze(1).expand(-1, num_keys).clone()
        idx_mask = torch.zeros(num_nodes, num_keys, dtype=torch.bool)
        for node_idx, keys in tqdm(
            enumerate(mapping),
            total=num_nodes,
            desc=f"SphereNeighborhood - index mapping {rank}",
        ):
            keys = list(keys)
            idx[node_idx, :len(keys)] = torch.tensor(keys)
            idx_mask[node_idx, :len(keys)] = True

        self.num_keys = num_keys
        self.register_buffer("idx", idx[None, None, :, :, None], persistent=False)
        self.register_buffer("idx_mask", idx_mask[None, None, :, :], persistent=False)
