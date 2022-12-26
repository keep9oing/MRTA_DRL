import ray
import torch

from runner import ModelRunner
from env.static_env import TA_Static
from utils import load_ray_config


# Load Ray Config
# Load Ray Config
ray_cfg = load_ray_config()

@ray.remote(num_gpus= ray_cfg["num_gpu"] / ray_cfg["num_test_worker"], num_cpus=1)
class TestWorker(object):
    def __init__(self, workerID, cfg, decode_type='greedy', plot=False):
        self.ID = workerID
        self.cfg = cfg

        self.decode_type = decode_type
        self.model = ModelRunner(self.cfg,self.decode_type, training=False)
        self.model.to(self.cfg["device"])
        self.local_decoder_gradient = []
        self.local_agent_encoder_gradient = []
        self.local_target_encoder_gradient = []

        self.plot = plot

    def run(self, env):
        return self.model(env)

    def set_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def sample(self, cfg, env_seed):
        env = TA_Static(cfg, seed_ext=env_seed)
        env.reset()
        with torch.no_grad():
            route_set, _, total_reward, _, max_flight_time, _, _ = self.run(env)
        if self.plot:
            return total_reward, route_set # use this code for plot
        else:
            return total_reward, max_flight_time
