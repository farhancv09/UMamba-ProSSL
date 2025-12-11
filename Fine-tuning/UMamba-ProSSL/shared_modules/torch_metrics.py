from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat
import torch
import picai_eval
from report_guided_annotation import extract_lesion_candidates

class PlaceholderMetrics:
    def __init__(self):
        self.score = 0
        self.auroc = 0
        self.AP = 0

class PicaiMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")

    def update(self, preds, target) -> None:
        self.preds.append(preds)
        self.target.append(target)

    def compute(self):
        preds = dim_zero_cat(self.preds).cpu().numpy()
        target = dim_zero_cat(self.target).cpu().numpy()
        
        metrics = picai_eval.evaluate(
            y_det=preds,
            y_true=target,
            y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0],
            y_true_postprocess_func=lambda y: y,
            num_parallel_calls=90
        )

        return metrics