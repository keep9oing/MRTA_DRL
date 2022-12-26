import yaml
import os
import shutil

def load_train_config(config_name):
    assert config_name is not None, "Name of configuration file should be defined"
    config_path = "config/"+config_name+".yaml"
    if not os.path.exists(config_path):
        raise ValueError("There is no {}".format(config_path))
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def load_model_config(config_name, exp_num):
    assert config_name is not None, "Name of configuration file should be defined"
    config_path = "model_save/"+config_name+'/'+"exp_{}_cfg.yaml".format(exp_num)
    if not os.path.exists(config_path):
        raise ValueError("There is no {}".format(config_path))
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg

def load_ray_config():
    # Load Ray Config
    ray_config_path = "config/ray.yaml"
    if not os.path.exists(ray_config_path):
        raise ValueError("There is no {}".format(ray_config_path))
    with open(ray_config_path, 'r') as f:
        ray_cfg = yaml.safe_load(f)

    return ray_cfg

def save_configs(config_name, model_path):
    config_path = "config/"+config_name+".yaml"
    if not os.path.exists(config_path):
        raise ValueError("There is no {}".format(config_path))
    shutil.copy(config_path, model_path+"_cfg.yaml")
    shutil.copy("config/ray.yaml", model_path+"_ray_cfg.yaml")
