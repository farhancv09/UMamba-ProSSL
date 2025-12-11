import json
import pandas as pd
import yaml
import munch
import numpy
import torch
pd.options.mode.chained_assignment = None  # default='warn'

def load_config(fn: str='config.yaml'):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, 'r') as stream:
        config=yaml.safe_load(stream)
    return munch.munchify(config)


def load_pretrained_model(model):
    store_dict = model.state_dict()
    pretrained_path = "trained_models/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    model_dict = torch.load(pretrained_path)["state_dict"]


    for key in model_dict.keys():
        if "out" not in key:
            store_dict[key].copy_(model_dict[key])

    model.load_state_dict(store_dict)
    print("Using pretrained weights!")
    return model