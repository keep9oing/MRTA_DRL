from env.static_env import TA_Static
from utils import load_train_config

config_name = "simple_het"

cfg = load_train_config(config_name)

seed=[0,1,2,3,4]

envs = [TA_Static(cfg, i) for i in seed]

for e in envs:
  e.reset()
  print(e.vehicle_initial)
