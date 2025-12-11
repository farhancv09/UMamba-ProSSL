from typing import Literal

from torch import nn
from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet

from nnssl.architectures.noskipResEncUNet import ResidualEncoderUNet_noskip
from nnssl.nets.UMambaBot import UMambaBot


SUPPORTED_ARCHITECTURES = Literal[
    "ResEncL",
    "NoSkipResEncL",
    "PrimusS",
    "PrimusB",
    "PrimusM",
    "PrimusL",
    "UMambaBot",
]
PRIMUS_SCALES = Literal["S", "M", "B", "L"]


def get_res_enc_l(
    num_input_channels: int, num_output_channels: int, deep_supervision: bool = False
) -> ResidualEncoderUNet:
    """
    Creates the ResEnc-L architecture used in "Revisiting MAE Pre-training ..."
    https://arxiv.org/abs/2410.23132
    """
    n_stages = 6
    network = ResidualEncoderUNet(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(n_stages)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
    return network


def get_noskip_res_enc_l(num_input_channels: int, num_output_channels: int) -> ResidualEncoderUNet:
    """
    Creates the ResEnc-L architecture used in "Revisiting MAE Pre-training ..."
    https://arxiv.org/abs/2410.23132
    """
    network = ResidualEncoderUNet_noskip(
        input_channels=num_input_channels,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
        kernel_sizes=[[3, 3, 3] for _ in range(6)],
        strides=[[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=[1, 1, 1, 1, 1],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    return network


def get_umamba_bot(
    num_input_channels: int,
    num_output_channels: int,
    *,
    n_stages: int = 7,
    features_per_stage: list[int] | None = None,
    kernel_sizes: list[tuple[int, int, int]] | None = None,
    strides: list[tuple[int, int, int]] | None = None,
    n_conv_per_stage: list[int] | None = None,
    n_conv_per_stage_decoder: list[int] | None = None,
    deep_supervision: bool = False,
) -> UMambaBot:
    """
    Helper used by get_network_by_name to instantiate UMambaBot with sensible defaults.
    """
    if features_per_stage is None:
        features_per_stage = [32, 64, 128, 256, 256, 384, 512]
    if kernel_sizes is None:
        kernel_sizes = [(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)]
    if strides is None:
        strides = [(1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)]
    if n_conv_per_stage is None:
        n_conv_per_stage = [2, 2, 2, 2, 2, 2, 2]
    if n_conv_per_stage_decoder is None:
        n_conv_per_stage_decoder = [2, 2, 2, 2, 2, 2]

    conv_op = nn.Conv3d
    norm_op = nn.InstanceNorm3d

    return UMambaBot(
        input_channels=num_input_channels,
        n_stages=n_stages,
        features_per_stage=features_per_stage,
        conv_op=conv_op,
        kernel_sizes=kernel_sizes,
        strides=strides,
        n_conv_per_stage=n_conv_per_stage,
        num_classes=num_output_channels,
        n_conv_per_stage_decoder=n_conv_per_stage_decoder,
        conv_bias=True,
        norm_op=norm_op,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=deep_supervision,
    )
