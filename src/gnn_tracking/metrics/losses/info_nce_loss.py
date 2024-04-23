from __future__ import print_function

import torch
import torch.nn as nn
from torch import Tensor as T
from torch.linalg import norm
from pytorch_lightning.core.mixins.hparams_mixin import HyperparametersMixin
from torch_cluster import radius_graph

from gnn_tracking.utils.graph_masks import get_good_node_mask_tensors
from gnn_tracking.metrics.losses import MultiLossFct, MultiLossFctReturn

def _InfoNCE_loss(
    *,
    x: T,
    att_edges: T,
    rep_edges: T,
    sim_func: str,
    tau: float
):
    if sim_func == 'l2_rbf':
       sigma = 0.75
       # distance between `attractive_edges`
       l2_dist_1 = torch.linalg.norm(x[att_edges[0]] - x[att_edges[1]], ord=2, dim=-1)
       # distance between `repulsive_edges`
       l2_dist_2 = torch.linalg.norm(x[rep_edges[0]] - x[rep_edges[1]], ord=2, dim=-1)
       similarity_1 = torch.exp(-l2_dist_1 / (2 * sigma**2))
       similarity_2 = torch.exp(-l2_dist_2 / (2 * sigma**2))
       
       max_sim_1 = (similarity_1 / tau).max()
       max_sim_2 = (similarity_2 / tau).max()
       exp_sim_1 = torch.exp(similarity_1 / tau - max_sim_1)
       exp_sim_2 = torch.exp(similarity_2 / tau - max_sim_2)

    num = exp_sim_1
    denom = num + torch.sum(exp_sim_2)
    return torch.mean(-torch.log(num / denom))


class InfoNCELoss(MultiLossFct, HyperparametersMixin):
    def __init__(self,
                tau: float = 0.05,
                dist_metric: str = 'l2_rbf',
                r_emb: float = 1.0,
                max_num_neighbors: int = 256,
                pt_thld: float = 0.9,
                max_eta: float = 4.0,
                rep_oi_only: bool = True,
                ):
        super(InfoNCELoss, self).__init__()
        self.save_hyperparameters()

    def _get_edges(
        self, *, x: T, batch: T, true_edge_index: T, mask: T, particle_id: T
    ) -> tuple[T, T]:
        """Returns edge index for graph"""
        near_edges = radius_graph(
            x,
            r=self.hparams.r_emb,
            batch=batch,
            loop=False,
            max_num_neighbors=self.hparams.max_num_neighbors,
        )
        # Every edge has to start at a particle of interest, so no special
        # case with noise
        if self.hparams.rep_oi_only:
            rep_edges = near_edges[:, mask[near_edges[0]]]
        else:
            rep_edges = near_edges
        rep_edges = rep_edges[:, particle_id[rep_edges[0]] != particle_id[rep_edges[1]]]
        att_edges = true_edge_index[:, mask[true_edge_index[0]]]
        return att_edges, rep_edges

    def forward(
        self,
        *,
        x: T,
        particle_id: T,
        batch: T,
        true_edge_index: T,
        pt: T,
        eta: T,
        reconstructable: T,
        **kwargs,
    ) -> MultiLossFctReturn:
        if true_edge_index is None:
            msg = (
                "True_edge_index must be given and not be None. Are you trying to use "
                "this loss for OC training? In this case, double check that you are "
                "properly passing on the true edges."
            )
            raise ValueError(msg)
        mask = get_good_node_mask_tensors(
            pt=pt,
            particle_id=particle_id,
            reconstructable=reconstructable,
            eta=eta,
            pt_thld=self.hparams.pt_thld,
            max_eta=self.hparams.max_eta,
        )
        # oi = of interest
        n_hits_oi = mask.sum()
        att_edges, rep_edges = self._get_edges(
            x=x,
            batch=batch,
            true_edge_index=true_edge_index,
            mask=mask,
            particle_id=particle_id,
        )
        info_loss = _InfoNCE_loss(
            x=x,
            att_edges=att_edges,
            rep_edges=rep_edges,
            sim_func=self.hparams.dist_metric,
            tau=self.hparams.tau
        )
        
        losses = {
            "attractive": info_loss,
            "repulsive": 0
        }
        weights: dict[str, float] = {
            "attractive": 1.0,
            "repulsive": 0,
        }
        extra = {
            "n_hits_oi": n_hits_oi,
            "n_edges_att": att_edges.shape[1],
            "n_edges_rep": rep_edges.shape[1],
        }
        return MultiLossFctReturn(
            loss_dct=losses,
            weight_dct=weights,
            extra_metrics=extra,
        )

class SLw(nn.Module, HyperparametersMixin):
    def __init__(
            self,
            temperature: float
        ):
        super().__init__()
        self._loss_fct = SupConLoss()
        self.temperature = temperature
        self.save_hyperparameters()

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor
    ) -> torch.Tensor:
        return self._loss_fct(input, target)