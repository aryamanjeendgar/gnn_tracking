"""Lightning module for edge classifier training."""

from typing import Any

from torch import Tensor
from torch import Tensor as T
from torch import nn
from torch_geometric.data import Data

from gnn_tracking.metrics.binary_classification import (
    get_maximized_bcs,
    get_roc_auc_scores,
)
from gnn_tracking.training.base import TrackingModule
from gnn_tracking.utils.lightning import obj_from_or_to_hparams
from gnn_tracking.utils.oom import tolerate_some_oom_errors


class ECModule(TrackingModule):
    def __init__(
        self,
        *,
        loss_fct: nn.Module,
        **kwargs,
    ):
        """Lightning module for edge classifier training."""
        super().__init__(**kwargs)
        self.loss_fct = obj_from_or_to_hparams(self, "loss_fct", loss_fct)

    def get_losses(self, out: dict[str, Any], data: Data) -> T:
        return self.loss_fct(
            w=out["W"],
            y=data.y.float(),
            pt=data.pt,
        )

    @tolerate_some_oom_errors
    def training_step(self, batch: Data, batch_idx: int) -> Tensor | None:
        batch = self.data_preproc(batch)
        out = self(batch)
        loss = self.get_losses(out, batch)
        self.log("total_train", loss.float(), prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch: Data, batch_idx: int):
        batch = self.data_preproc(batch)
        out = self(batch)
        loss = self.get_losses(out, batch)
        self.log("total", loss.float(), on_epoch=True)
        metrics = {}
        # todo: this needs to be done for different pt thresholds
        metrics |= get_roc_auc_scores(
            true=batch.y, predicted=out["W"], max_fprs=[None, 0.01, 0.001]
        )
        metrics |= get_maximized_bcs(y=batch.y, output=out["W"])
        # todo: add graph analysis
        self.log_dict_with_errors(
            dict(sorted(metrics.items())),
            batch_size=self.trainer.val_dataloaders.batch_size,
        )

    def highlight_metric(self, metric: str) -> bool:
        return "max_mcc" in metric