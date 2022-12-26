import ray
import torch

from runner import ModelRunner
from env.static_env import TA_Static
from utils import load_ray_config

# Load Ray Config
ray_cfg = load_ray_config()

@ray.remote(num_gpus= ray_cfg["num_gpu"]/ray_cfg["num_worker"], num_cpus=1)
class Worker(object):
    def __init__(self, workerID, cfg, decode_type='sampling'):
        self.ID = workerID
        self.cfg = cfg
        self.device = self.cfg["device"]

        self.model = ModelRunner(self.cfg)
        self.baseline_model = ModelRunner(self.cfg, decode_type='greedy')
        self.model.to(self.device)
        self.baseline_model.to(self.device)
        self.local_model_gradient = []

        self.reward_buffer = []
        self.max_flight_time_buffer = []
        self.total_reward_buffer = []
        self.baseline_buffer = []
        self.episode_buffer = []
        for i in range(5):
            self.episode_buffer.append([])

        self.decode_type = decode_type

        self.env = TA_Static(self.cfg)

    def run_model(self, env):
        return self.model(env)

    def run_baseline(self, env):
        return self.baseline_model(env)

    def get_logp(self):
        agent_inputs = torch.cat(self.episode_buffer[0]).squeeze(0).to(self.device)
        idle_embeddings = torch.cat(self.episode_buffer[1]).squeeze(0).to(self.device)
        task_inputs = torch.cat(self.episode_buffer[2]).squeeze(0).to(self.device)
        mask = torch.cat(self.episode_buffer[3]).squeeze(0).to(self.device)
        agent_feature = self.model.local_agent_encoder(agent_inputs)
        target_feature = self.model.local_target_encoder(task_inputs, idle_embeddings)
        _, log_prob = self.model.local_decoder(target_feature=target_feature,
                                                   current_state=torch.mean(target_feature,dim=1).unsqueeze(1),
                                                   agent_feature=agent_feature,
                                                   mask=mask,
                                                   decode_type=self.decode_type)
        action_list=torch.cat(self.episode_buffer[4]).squeeze(0).to(self.device)
        logp=torch.gather(log_prob,1,action_list.unsqueeze(1))
        entropy=(log_prob*log_prob.exp()).sum(dim=-1).mean()
        return logp, entropy

    def get_advantage(self, reward_buffer, baseline):
        advantage = (reward_buffer - baseline)
        return advantage

    def get_loss(self, advantage, log_p_buffer, entropy_buffer):
        policy_loss = -log_p_buffer.squeeze(1) * advantage.detach()
        loss = policy_loss.sum()/ray_cfg["epi_per_worker"]
        return loss

    def get_gradient(self, loss):
        self.model.zero_grad()
        loss.backward()
        g = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1000, norm_type=2)
        self.local_model_gradient = []
        for local_param in self.model.parameters():
            self.local_model_gradient.append(local_param.grad)
        return g

    def set_model_weights(self, global_weights):
        self.model.load_state_dict(global_weights)

    def set_baseline_model_weights(self, baseline_weights):
        self.baseline_model.load_state_dict(baseline_weights)

    def sample(self):
        self.env.reset()

        with torch.no_grad():
            route_set, reward, total_reward, max_flight_time_ngative, max_flight_time, max_id, episode_buffer = self.run_model(self.env)

        self.env.generate_mask()

        with torch.no_grad():
            base_route, base_reward, base_total_reward, baseline_max_flight_time_negative, base_max_flight_time, base_max_id, base_episode_buffer = self.run_baseline(self.env)

        self.reward_buffer += reward
        self.total_reward_buffer.append(total_reward)
        self.max_flight_time_buffer.append(max_flight_time)
        self.baseline_buffer += baseline_max_flight_time_negative.expand_as(reward)
        for i in range(5):
            self.episode_buffer[i] += episode_buffer[i]

    def return_gradient(self):
        reward_buffer = torch.stack(self.reward_buffer)
        log_p_buffer, entropy_loss = self.get_logp()
        baseline_buffer = torch.stack(self.baseline_buffer)
        advantage = self.get_advantage(reward_buffer=reward_buffer, baseline=baseline_buffer)
        loss = self.get_loss(advantage, log_p_buffer, entropy_loss)
        grad_norm = self.get_gradient(loss)
        max_flight_time = torch.stack(self.max_flight_time_buffer).squeeze(0).mean()
        total_reward = torch.stack(self.total_reward_buffer).squeeze(0).mean()

        self.reward_buffer = []
        self.total_reward_buffer = []
        self.episode_buffer = []
        for i in range(5):
            self.episode_buffer.append([])
        self.max_flight_time_buffer = []
        self.baseline_buffer = []

        # Random Tasks & Vehicles
        return self.local_model_gradient, loss.mean().item(), grad_norm, advantage.mean().item(), max_flight_time.item(), entropy_loss.mean().item(), total_reward.item()


if __name__ == '__main__':
    import yaml
    import os

    config_name = "simple"

    assert config_name is not None, "Name of configuration file should be defined"

    config_path = "config/"+config_name+".yaml"
    if not os.path.exists(config_path):
        raise ValueError("There is no {}".format(config_path))
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # env = TA_Static(cfg)

    worker = Worker(1,cfg)

    for i in range(4):
        worker.sample()

    for i in range(5):
        print(torch.cat(worker.episode_buffer[i]).squeeze(0).size())

    worker.return_gradient()

    worker = Worker(1,cfg)

    for i in range(4):
        worker.sample()

    for i in range(5):
        print(torch.cat(worker.episode_buffer[i]).squeeze(0).size())
