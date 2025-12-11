from monai import transforms

def get_post_transforms(key, 
                        orig_key, 
                        orig_transforms, 
                        output_dtype="uint8", 
                        save_mask=True, 
                        out_dir="outputs/",
                        keep_n_largest_components=1, 
                        output_postfix="pred"):
    post_transforms = [
            transforms.Invertd(
                keys=[key],
                transform=orig_transforms,
                orig_keys=[orig_key],
                nearest_interp=False,
                to_tensor=True,
            )
        ]
    if keep_n_largest_components > 0:
        post_transforms.append(transforms.KeepLargestConnectedComponentd(keys=[key], num_components=keep_n_largest_components))

    if save_mask:
        post_transforms = [
                *post_transforms,
                transforms.SaveImaged(
                    keys=[key],
                    output_dir=out_dir,
                    output_postfix=output_postfix,
                    output_dtype=output_dtype,
                    resample=False,
                    separate_folder=False
                )  
        ]

    return transforms.Compose(post_transforms)