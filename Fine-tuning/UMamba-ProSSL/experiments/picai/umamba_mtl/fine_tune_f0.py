import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from functools import partial
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
import warnings
import sys
from torch import nn
repo_root = ""
sys.path.append(repo_root)
from shared_modules.data_module import DataModule
from shared_modules.module_selector import get_network, get_logger, get_callbacks, get_loss_fn
from shared_modules.utils import load_config
from shared_modules.plotting import plot_confusion, plot_difference, plot_metrics
from monai.optimizers.lr_scheduler import WarmupCosineSchedule
from shared_modules.torch_metrics import PicaiMetric


torch.set_float32_matmul_precision("high")
warnings.filterwarnings("ignore", category=UserWarning)



class LitModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lr = config.lr

        self.model = get_network(config)
        self.loss_fn_pca = get_loss_fn(config)
        self.loss_fn_anatomy = DiceCELoss(include_background=True, sigmoid=True, weight=torch.tensor([1, 1, 1, 1]))
        self.dsc_fn = DiceMetric(include_background=True, reduction="mean")
        self.picai_metric_fn = PicaiMetric()

        self.inferer = partial(
            sliding_window_inference,
            roi_size=config.transforms.spatial_size,
            sw_batch_size=config.sw_batch_size,
            predictor=self.model,
            overlap=config.infer_overlap,
        )
        self.validation_step_outputs = {'dsc': []}



    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        lr_scheduler = WarmupCosineSchedule(optimizer=optimizer, warmup_steps=self.config.warmup_epochs, t_total=self.config.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        img, pca, zones = batch["image"], batch["pca"], batch["zones"]
        prostate = torch.logical_or(zones[:,1:2,...], zones[:,2:3,...]).float()

        logits = self.inferer(img) 
    
        logits_pca = logits[:,1:2,...]  
        logits_zonal = logits[:,2:,...] 
        logits_prostate = logits_zonal[:,1:,...].sum(dim=1, keepdim=True) # sum of PZ and TZ
        logits_anatomy = torch.cat([logits_zonal,logits_prostate], dim=1) 
        
        gt_anatomy = torch.cat([zones,prostate], dim=1)
        
        loss_anatomy = self.loss_fn_anatomy(logits_anatomy, gt_anatomy) 
        loss_pca = self.loss_fn_pca(logits_pca, pca)
        
        loss = 0.2 * loss_anatomy + loss_pca
        
        
        self.log_dict({
            "train/loss": loss,
        }, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        img, pca, zones = batch["image"], batch["pca"], batch["zones"]
        prostate = torch.logical_or(zones[:,1:2,...], zones[:,2:3,...]).float()
        
        logits = self.inferer(img) 
    
        logits_pca = logits[:,1:2,...]  
        tumor_probability = torch.sigmoid(logits_pca[0,0])[None]
        tumor_pred = (tumor_probability >= 0.5).float()
        
        logits_zonal = logits[:,2:,...] 
        zonal_probability = torch.softmax(logits_zonal, dim=1)
        prostate_probability = zonal_probability[:,1:, ...].sum(dim=1, keepdim=True)
        anatomy_pred = (torch.cat([zonal_probability,prostate_probability], dim=1) >= 0.5).float()

        pred_all = torch.cat([tumor_pred[None], anatomy_pred[:,1:,...]], dim=1)
        gt_all = torch.cat([pca, zones[:,1:,...], prostate], dim=1)

        self.dsc_fn(y_pred=pred_all, y=gt_all)
        self.picai_metric_fn.update(preds=tumor_probability, target=pca[0])
    
    def on_validation_epoch_end(self):
        dsc = self.dsc_fn.aggregate("mean_batch")
        self.dsc_fn.reset()
        metrics = self.picai_metric_fn.compute()
        # print(metrics)

        # if self.current_epoch < 15:
        #     metrics = PlaceholderMetrics()
        
        if self.global_rank == 0:
        
            images = [plot_fn(metrics, epoch=self.current_epoch) for plot_fn in [plot_metrics, plot_confusion, plot_difference]]
            self.logger.log_image(key=f"rank_{self.global_rank}", images=images)      
                  
        self.log_dict({
                "val/pi_cai_score": metrics.score,
                "val/ap": metrics.AP,
                "val/auroc": metrics.auroc,
                "val/dsc_tumor": dsc[0],
                "val/dsc_pz": dsc[1],
                "val/dsc_tz": dsc[2],
                "val/dsc_prostate": dsc[3],
            }, sync_dist=True, prog_bar=True, on_epoch=True)  

        self.picai_metric_fn.reset()



if __name__ == "__main__":
    config = load_config("") # yaml file path

    data_module = DataModule(
        config,
    )

    logger = get_logger(config)
    model = LitModel(config)
    

    if config.checkpoint is not None and not config.test_mode:
        print("Fine tuning model from checkpoint:", config.checkpoint)
        checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
        state_dict = checkpoint["network_weights"] if "network_weights" in checkpoint else checkpoint
        model.model.load_state_dict(state_dict, strict=False)
        # do this for MAE and MG and not for VF
        new_layers = nn.ModuleList()
        for layer in model.model.decoder.seg_layers:
            # print(layer)
            if isinstance(layer, nn.Conv3d):
                new_layers.append(
                    nn.Conv3d(
                        in_channels=layer.in_channels,
                        out_channels=5,
                        kernel_size=layer.kernel_size,
                        stride=layer.stride,
                        padding=layer.padding,
                        bias=(layer.bias is not None)
                    )
                )
            else:
                new_layers.append(layer)
        model.model.decoder.seg_layers = new_layers

    
    trainer = Trainer(
        accelerator="gpu",
        fast_dev_run=config.fast_dev_run,
        devices=config.gpus,
        strategy="auto",
        precision="16",
        num_sanity_val_steps=0,
        max_epochs=config.max_epochs,
        check_val_every_n_epoch=config.val_every,
        enable_checkpointing=config.save_checkpoint,
        gradient_clip_val=1.0,
        logger=logger,
        # use_distributed_sampler=False,
        callbacks=get_callbacks(config, logger),
    )
    
    if config.test_mode:
        trainer.test(model, data_module, ckpt_path=config.checkpoint)
    else:
        if config.resume_ckpt:
            trainer.fit(model, data_module, ckpt_path=config.checkpoint)
        else:
            trainer.fit(model, data_module)
