import os
from typing import override
import torch
from torch import nn

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.get_network_by_name import get_network_by_name
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.dataloading.model_genesis_transform import ModelGenesisTransform
from nnssl.training.loss.mse_loss import LossMaskMSELoss
from nnssl.data.nnsslFilter.modality_filter import ModalityFilter
from dynamic_network_architectures.building_blocks.helper import convert_dim_to_conv_op, get_matching_instancenorm
from nnssl.nets.UMambaBot import UMambaBot

from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from torch import autocast
from nnssl.utilities.helpers import dummy_context

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import save_json


class ModelGenesisTrainer(AbstractBaseTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
        save_batch: bool =True,
    ):
        super(ModelGenesisTrainer, self).__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (20, 256, 256)
        self.save_batch = save_batch
        self._saved_batch_idx = 0
        if self.save_batch:
            self.saved_batch_dir = os.path.join(self.output_folder, "saved_batches")
            os.makedirs(self.saved_batch_dir, exist_ok=True)
        else:
            self.saved_batch_dir = None
        self.num_epochs = 1200

    def build_loss(self):
        """
        This is where you build your loss function. You can use anything from torch.nn here.
        In general the MAE losses are only applied on regions where the mask is 0.

        :return:
        """
        return torch.nn.MSELoss()

    @override
    def build_architecture_and_adaptation_plan(
        self, config_plan: ConfigurationPlan, num_input_channels: int, num_output_channels: int
    ) -> nn.Module:
        network = get_network_by_name(
            config_plan,
            "ResEncL",
            num_input_channels,
            num_output_channels,
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("ResEncL"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=1,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.convs.0.conv", "encoder.stem.convs.0.all_modules.0"),
        )
        return network, adapt_plan

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically."""
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size=patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    def train_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.permute(0, 1, 3, 4, 2).contiguous()
        target  = target.permute(0, 1, 3, 4, 2).contiguous()
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast is a little bitch.
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(in_data)
            # del data
            l = self.loss(target, output)

        if self.save_batch:
            out_path = os.path.join(
                self.saved_batch_dir, f"train_step_ep{self.current_epoch}_idx{self._saved_batch_idx}.pt"
            )
            torch.save(
                {"input": in_data.detach().cpu(), "target": target.detach().cpu(), "output": output.detach().cpu()},
                out_path,
            )
            self._saved_batch_idx += 1

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        breakpoint()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        in_data = in_data.permute(0, 1, 3, 4, 2).contiguous()
        target  = target.permute(0, 1, 3, 4, 2).contiguous()
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(in_data)
                # del data
                l = self.loss(target, output)

        return {"loss": l.detach().cpu().numpy()}

    @staticmethod
    def get_training_transforms() -> AbstractTransform:
        tr_transforms = []

        tr_transforms.append(ModelGenesisTransform())
        tr_transforms.append(NumpyToTensor(["input", "target"], "float"))
        tr_transforms.append(NumpyToTensor(["seg"], "long"))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        return ModelGenesisTrainer.get_training_transforms()


class ModelGenesisTrainerUMambaBot(ModelGenesisTrainer):
    """
    ModelGenesis pre-training with a custom-configured UMambaBot encoder/decoder.
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (20, 256, 256)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 2
        self.num_input_channels = 3
        

    @override
    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> nn.Module:
        # Reconstruction task: force decoder output channels to match inputs.
        num_output_channels = num_input_channels
        conv_op = convert_dim_to_conv_op(3)
        norm_op = get_matching_instancenorm(conv_op)
        architecture = UMambaBot(
            input_channels=num_input_channels,
            n_stages=7,
            features_per_stage=[32, 64, 128, 256, 256, 384, 512],
            conv_op=conv_op,
            kernel_sizes=[(1, 3, 3), (1, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            strides=[(1, 1, 1), (1, 2, 2), (1, 2, 2), (2, 2, 2), (2, 2, 2), (1, 2, 2), (1, 2, 2)],
            n_conv_per_stage=[2, 2, 2, 2, 2, 2, 2],
            num_classes=num_output_channels,
            n_conv_per_stage_decoder=[2, 2, 2, 2, 2, 2],
            conv_bias=True,
            norm_op=norm_op,
            norm_op_kwargs={"eps": 1e-5, "affine": True},
            dropout_op=None,
            dropout_op_kwargs=None,
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={"inplace": True},
            deep_supervision=False,
        )
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("UMambaBot"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=num_input_channels,
            key_to_encoder="encoder.stages",
            key_to_stem="encoder.stem",
            keys_to_in_proj=("encoder.stem.0.conv1",),
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        print("UMamba trainer")
        return architecture, adapt_plan


class ModelGenesisTrainer_BS8(ModelGenesisTrainer):

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class ModelGenesisTrainer_ANAT(ModelGenesisTrainer):

    def get_dataloaders(self):
        """
        Dataloader creation is very different depending on the use-case of training.
        This method has to be implemneted for other use-cases aside from MAE more specifically."""
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.config_plan.patch_size

        tr_transforms = self.get_training_transforms()
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_foreground_dataloaders(initial_patch_size=patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val


class ModelGenesisTrainer_ANON(ModelGenesisTrainer):

    def build_loss(self):
        return LossMaskMSELoss()

    def train_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        anon = batch["seg"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        loss_mask = 1 - anon

        self.optimizer.zero_grad(set_to_none=True)
        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            output = self.network(in_data)
            l = self.loss(output, target, loss_mask)

        if self.save_batch:
            out_path = os.path.join(
                self.saved_batch_dir, f"train_step_ep{self.current_epoch}_idx{self._saved_batch_idx}.pt"
            )
            torch.save(
                {"input": in_data.detach().cpu(), "target": target.detach().cpu(), "output": output.detach().cpu()},
                out_path,
            )
            self._saved_batch_idx += 1

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        in_data = batch["input"]
        target = batch["target"]
        anon = batch["seg"]
        in_data = in_data.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        anon = anon.to(self.device, non_blocking=True)

        loss_mask = 1 - anon

        with torch.no_grad():
            with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
                output = self.network(in_data)
                l = self.loss(output, target, loss_mask)

        return {"loss": l.detach().cpu().numpy()}


class ModelGenesisTrainer_ANAT_ANON(ModelGenesisTrainer_ANAT, ModelGenesisTrainer_ANON):
    pass


class ModelGenesisTrainer_ANAT_ANON_test(ModelGenesisTrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].batch_size = 2
        plan.configurations[configuration_name].patch_size = (96, 96, 96)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)


class ModelGenesisTrainer_ANAT_ANON_BS8(ModelGenesisTrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8


class ModelGenesisTrainer_ANAT_ANON_BS8_T1w_T2w_FLAIR(ModelGenesisTrainer_ANAT_ANON):
    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        plan.configurations[configuration_name].patch_size = (160, 160, 160)
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.total_batch_size = 8
        self.iimg_filters.append(ModalityFilter(valid_modalities=["T1w", "T2w", "FLAIR"]))
