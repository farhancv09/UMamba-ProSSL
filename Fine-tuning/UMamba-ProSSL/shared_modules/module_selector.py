from monai.networks.nets import SwinUNETR
from monai.losses import FocalLoss, DiceFocalLoss, TverskyLoss, DeepSupervisionLoss
from shared_modules.networks.UMambaBot_3d import UMambaBot
from shared_modules.networks.UMambaBot_3d_mtl import UMambaBotMTL
import wandb
from lightning.pytorch.loggers import WandbLogger
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

def get_network(config):
    if config.network.name == "SwinUNETR":
        model = SwinUNETR(
        img_size=config.network.img_size,
        in_channels=config.network.in_channels,
        out_channels=config.network.out_channels,
        feature_size=config.network.feature_size,
        drop_rate=config.network.drop_rate,
        attn_drop_rate=0.0,
        dropout_path_rate=config.network.dropout_path_rate,
        use_checkpoint=True,
        use_v2=config.network.use_v2
    )
    elif config.network.name == "UMambaBot":
        model = UMambaBot(
            input_channels=config.network.in_channels,
            n_stages=config.network.n_stages,
            features_per_stage=config.network.features_per_stage,
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=config.network.kernel_sizes,
            num_classes=config.network.num_classes,
            n_conv_per_stage_decoder=config.network.n_conv_per_stage_decoder,
            n_conv_per_stage=config.network.n_conv_per_stage,
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-05, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.modules.activation.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            strides=config.network.strides,
            deep_supervision=config.network.deep_supervision
        )
    elif config.network.name == "UMambaBotMTL":
        model = UMambaBotMTL(
            input_channels=config.network.in_channels,
            n_stages=config.network.n_stages,
            features_per_stage=config.network.features_per_stage,
            conv_op=torch.nn.modules.conv.Conv3d,
            kernel_sizes=config.network.kernel_sizes,
            num_classes=config.network.num_classes,
            n_conv_per_stage_decoder=config.network.n_conv_per_stage_decoder,
            n_conv_per_stage=config.network.n_conv_per_stage,
            conv_bias=True,
            norm_op=torch.nn.modules.instancenorm.InstanceNorm3d,
            norm_op_kwargs={'eps': 1e-05, 'affine': True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=torch.nn.modules.activation.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            strides=config.network.strides,
            deep_supervision=config.network.deep_supervision
        )
    else:
        raise Exception(f"{config.network.name} is not a valid network option! Chose one of [SwinUNETR, UMambaBot, UMambaBotMTL]")
    return model

def get_loss_fn(config):
    if config.loss.name == "FocalLoss":
        loss_fn = FocalLoss(include_background=True, to_onehot_y=True, gamma=config.loss.gamma)
    elif config.loss.name == "DiceFocalLoss":
        # loss_fn = DiceFocalLoss(include_background=True, to_onehot_y=True, softmax=True, gamma=config.loss.gamma)
        loss_fn = DiceFocalLoss(include_background=True, sigmoid=True, gamma=config.loss.gamma, lambda_dice=config.loss.lambda_dice, weight=config.loss.weight) #, weight=0.75
    elif config.loss.name == "TverskyLoss":
        loss_fn = TverskyLoss(include_background=True, sigmoid=True, alpha=config.loss.alpha, beta=config.loss.beta)
    else:
        raise Exception(f"{config.loss.name} is not a valid loss option! Chose one of [FocalLoss, DiceFocalLoss, TverskyLoss]")
    return loss_fn

    
def get_logger(config):
    wandb.config = config
    if not config.logger.active or config.test_mode:
        wandb.init(mode="disabled")

    config = config.logger
    wandb_logger = WandbLogger(
        name=config.experiment_name,
        version=config.resume_wandb_id,
        project=config.project,
        entity=config.entity,
        config=wandb.config,
        save_dir=f"/cluster/home/syedfa/SSL_study/fine_tuning/U-MambaMTL/experiments/picai/umamba/trained_models/{config.experiment_name}"
    )
    for metric in config.metrics:
        wandb_logger.experiment.define_metric(f"val/{metric}", summary="max")
        
    return wandb_logger
        

def get_callbacks(config, logger):
        callbacks = [
            EarlyStopping(monitor=f"val/{config.logger.metrics[0]}", mode="max", patience=config.early_stopping_patience, verbose=True),
            LearningRateMonitor(logging_interval="epoch"),
        ]
        if config.save_checkpoint:
            callbacks.append(ModelCheckpoint(
                dirpath=f"/cluster/home/syedfa/SSL_study/fine_tuning/U-MambaMTL/experiments/picai/umamba/trained_models/{config.logger.project}/{config.logger.experiment_name}/{config.network.name}/{logger.version}/",
                save_top_k=config.save_top_k,
                save_last=config.save_last,
                monitor=f"val/{config.logger.metrics[0]}",
                mode="max",
                filename="epoch={epoch:02d}-metric={val/metric:.4f}".replace("metric", config.logger.metrics[0] ),
                auto_insert_metric_name=False,
                save_weights_only=True
            ))
        return callbacks