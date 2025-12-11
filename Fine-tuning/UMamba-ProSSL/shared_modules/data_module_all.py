import pytorch_lightning as pl
from monai import data, transforms
from monai.data import load_decathlon_datalist, DataLoader
import torch

class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.json_list = config.data.json_list
        self.data_dir = config.data.data_dir
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.cache_rate = config.cache_rate
        self.config = config
        self.ds = {}
        self.transforms = {}
        self.image_keys = config.transforms.image_keys
        self.label_keys = config.transforms.label_keys
        self.all_keys = ["image"] + self.label_keys
        self.interpolation_modes_label = ["nearest" for _ in self.label_keys]
        self.interpolation_modes_all = ["bilinear"] + self.interpolation_modes_label

    def setup(self, stage: str):
        if stage == "test":
            splits = ["test"]
        else:
            splits = ["training", "validation"]

        for split in splits:
            datalist = self.get_datalist(split)
            self.transforms[split] = self.get_transforms(split) # transform objects are saved for inverse transforms

            self.ds[split] = data.CacheDataset(
                data=datalist,
                transform=self.transforms[split],
                cache_rate=self.config.cache_rate,# if split in ["validation", "test"] else 0,
                num_workers=self.num_workers,
            )

    def train_dataloader(self):
        return DataLoader(self.ds["training"], batch_size=self.batch_size, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.ds["validation"], batch_size=1, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.ds["test"], batch_size=1, pin_memory=True)
    
        
    def get_shared_transforms(self):
        to_discrete_keys = [key for key in self.label_keys if key != "zones"]
        
        if self.config.transforms.crop_key:
            crop_transforms = [
                    transforms.SpatialPadd(keys=self.all_keys, spatial_size=[128,128,20], mode="reflect"), # Option 1
                    transforms.CropForegroundd(keys=self.all_keys, 
                                            source_key=self.config.transforms.crop_key, 
                                            # margin=self.config.transforms.prostate_crop_margin,  # Option 1
                                            allow_smaller=False, 
                                            mode=self.config.transforms.padding_mode, 
                                            k_divisible=self.config.transforms.roi_size, # Option 2
                    ),
                    transforms.CenterSpatialCropd(keys=self.all_keys, roi_size=self.config.transforms.roi_size)
                    ]
        else:
            crop_transforms = [
                transforms.SpatialPadd(keys=self.all_keys, spatial_size=self.config.transforms.roi_size, mode=self.config.transforms.padding_mode),
                transforms.CenterSpatialCropd(keys=self.all_keys, roi_size=self.config.transforms.roi_size),
            ]
            
        if set(["t2w","adc", "hbv"]) == set(self.image_keys):
            """
            Use resample to match if each MRI sequence is provided seperatily. This is usefull in case of providing raw data.
            """
            
            load_img_transforms = [
                transforms.LoadImageD(keys="t2w", ensure_channel_first=True, reader="ITKReader"),
                transforms.LoadImageD(keys="adc", ensure_channel_first=True, reader="ITKReader"),
                transforms.LoadImageD(keys="hbv", ensure_channel_first=True, reader="ITKReader"),
                transforms.ResampleToMatchD(keys=["hbv", "adc"], key_dst="t2w"),
                transforms.SpacingD(keys=self.image_keys, pixdim=self.config.transforms.spacing),
                transforms.ConcatItemsd(self.image_keys, name="image"),
            ]
        else:
            load_img_transforms = [
                transforms.LoadImaged(keys=self.image_keys, ensure_channel_first=True),
                transforms.Spacingd(keys=self.image_keys, pixdim=self.config.transforms.spacing),
            ]
        if "zones" in self.label_keys:
            to_discrete_zones = [transforms.AsDiscreted(keys="zones", to_onehot=3, allow_missing_keys=True)] # split up the zones into channels]
        else:
            to_discrete_zones = []
        
            
        shared_transforms = [
                *load_img_transforms,
                transforms.LoadImaged(keys=self.label_keys, ensure_channel_first=True),
                transforms.Spacingd(keys=self.label_keys, pixdim=self.config.transforms.spacing, mode=self.interpolation_modes_label),
                transforms.AsDiscreted(keys=to_discrete_keys, threshold=0.5, allow_missing_keys=True), # In case of gleason grades
                *to_discrete_zones,
                transforms.Orientationd(keys=self.all_keys, axcodes="RAS"),
                *crop_transforms,
                transforms.ClipIntensityPercentilesd(keys=["image"], 
                                                        lower=self.config.transforms.intensity_clipping_lower, 
                                                        upper=self.config.transforms.intensity_clipping_upper, 
                                                        channel_wise=True),
                transforms.NormalizeIntensityd(keys=["image"], channel_wise=True),
        ]
            
        
        return shared_transforms

    def get_transforms(self, split):
        transform_dict = {}
        shared_transforms = self.get_shared_transforms()

        transform_dict["training"] = transforms.Compose(
            [
                *shared_transforms,
                transforms.RandFlipd(
                    keys=self.all_keys,
                    prob=self.config.transforms.RandFlipd_prob,
                    spatial_axis=0,
                ),
                transforms.RandFlipd(
                    keys=self.all_keys,
                    prob=self.config.transforms.RandFlipd_prob,
                    spatial_axis=1,
                ),
                transforms.RandFlipd(
                    keys=self.all_keys,
                    prob=self.config.transforms.RandFlipd_prob,
                    spatial_axis=2,
                ),
                transforms.RandGaussianSmoothd(
                keys="image", prob=self.config.transforms.RandGaussianSmoothd_prob, sigma_x=[0.5, 1.0], sigma_y=[0.5, 1.0], sigma_z=[0.5, 1.0]
                ),
                transforms.RandScaleIntensityd(
                    keys="image", factors=0.1, prob=self.config.transforms.RandScaleIntensityd_prob
                ),
                transforms.RandShiftIntensityd(
                    keys="image", offsets=0.1, prob=self.config.transforms.RandShiftIntensityd_prob
                ),
                transforms.RandGaussianNoised(keys="image", prob=self.config.transforms.RandGaussianNoised_prob, mean=0.0, std=0.1),
                transforms.RandAffined(
                keys=self.all_keys,
                prob=0.2,
                rotate_range=self.config.transforms.RandAffined_rotate_range,
                scale_range=self.config.transforms.RandAffined_scale_range,
                mode=self.interpolation_modes_all,
                cache_grid=True,
                spatial_size=self.config.transforms.roi_size,
                padding_mode=self.config.transforms.padding_mode,
            ),
                transforms.RandSpatialCropd(
                    keys=self.all_keys,
                    roi_size=self.config.transforms.spatial_size,
                ),
                
                transforms.ToTensord(keys=self.all_keys, track_meta=True),
            ]
        )

        transform_dict["validation"] = transforms.Compose(
            [
                *shared_transforms,
                transforms.ToTensord(keys=self.all_keys, track_meta=True),
            ]
        )

        transform_dict["test"] = transforms.Compose(
            [
                
                *shared_transforms,
            ]
        )
        

        return transform_dict[split]

    def get_datalist(self, split):
        datalist = load_decathlon_datalist(
            self.json_list, True, split, base_dir=self.data_dir
        )
        if torch.distributed.is_initialized():
            data_partition = data.DatasetFunc(
                data=datalist,
                func=lambda *config, **kwconfig: data.partition_dataset(*config, **kwconfig)[
                    torch.distributed.get_rank()
                ],
                num_partitions=torch.distributed.get_world_size(),
            )
            return data_partition
        else:
            return datalist
