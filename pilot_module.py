import torch
import torch.nn as nn

import copy

from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder


class Pilot(nn.Module):
    def __init__(self, pilot_cfg, type_name, type_info, id):
        super(Pilot, self).__init__()

        self.local_agent_encoder = AgentEncoder(pilot_cfg)
        self.local_target_encoder = TargetEncoder(pilot_cfg)
        self.local_decoder = Decoder(pilot_cfg)
        self.device = pilot_cfg["device"]

        self.local_agent_encoder.to(self.device)
        self.local_target_encoder.to(self.device)
        self.local_decoder.to(self.device)

        self.vehicle_type = type_name
        self.initial_vehicle_state = torch.from_numpy(type_info["position"][0,id,None,None]).to(self.device)
        self.vehicle_state = copy.deepcopy(self.initial_vehicle_state)
        self.angle = 0
        self.velocity = type_info["velocity"]


        self.task_list = torch.tensor([0], device=self.device)
        self.next_target = None
        self.next_action_gap = 0
        self.sum_distance = 0
        self.finish = False
        self.decode_type = "greedy"

    def get_initial_state(self):
        if self.device == "cpu":
            return self.initial_vehicle_state.numpy()
        else:
            return self.initial_vehicle_state.cpu().numpy()

    def get_vehicle_pos(self):
        if self.device == "cpu":
            return self.vehicle_state.numpy()
        else:
            return self.vehicle_state.cpu().numpy()

    def get_route(self):
        route = torch.cat((self.task_list[None,0],self.task_list[self.task_list!=0]))
        if self.device == "cpu":
            return route.numpy()
        else:
            return route.cpu().numpy()

    def get_target_set(self):
        if self.device == "cpu":
            return self.target_set.numpy()
        else:
            return self.target_set.cpu().numpy()

    def forward(self, obs):
        ## TODO: various fleet types
        obs["fleet"] = torch.from_numpy(obs["fleet"]).to(self.device)
        ## TODO: various task types
        obs["task"] = torch.from_numpy(obs["task"]["visit"]["position"]).to(self.device) ## This implementation is temporary
        obs["mask"] = torch.from_numpy(obs["mask"]).to(self.device)

        self.target_set = torch.cat((self.initial_vehicle_state, obs["task"]), dim=1)

        next_target_index, finish = self.select_next_target(obs)
        self.next_target = next_target_index.item()

        if finish is not True:
            self.update_state()  # use add_final_distance to add 'return to depot' distance
        else:
            self.next_action_gap = 0

        return next_target_index, finish

    def select_next_target(self, obs):
        if 0 in obs["mask"][0,:]:

            agent_feature, agent_input = self.calculate_encoded_agent(agent_inputs=obs["fleet"])
            task_feature, task_input, idle_embedding = self.calculate_encoded_target(task_inputs=obs["task"])

            mask = copy.deepcopy(obs["mask"])
            mask = torch.cat((torch.tensor([[0]], device=self.device),mask), dim=-1)

            next_target_index, log_prob = self.local_decoder(target_feature=task_feature,
                                                       current_state=torch.mean(task_feature,dim=1).unsqueeze(1),
                                                       agent_feature=agent_feature,
                                                       mask=mask,
                                                       decode_type=self.decode_type)
            self.task_list = torch.cat((self.task_list, next_target_index))
        else:
            self.finish = True
            next_target_index = None
        return next_target_index, self.finish

    def calculate_encoded_agent(self, agent_inputs):
        agent_inputs[0][:,:3] = agent_inputs[0][:,:3] - torch.cat((self.vehicle_state,torch.tensor([[[0]]], device=self.device)),dim=-1)
        agent_feature = self.local_agent_encoder(agent_inputs)
        return agent_feature, agent_inputs

    def calculate_encoded_target(self, task_inputs):
        task_set = copy.deepcopy(task_inputs)
        task_inputs = task_set - self.vehicle_state
        task_feature, idle_embedding = self.local_target_encoder(task_inputs)
        return task_feature, task_inputs, idle_embedding

    def update_state(self):
        if self.task_list[-1] != 0:
            route = torch.cat((self.task_list[None,0],self.task_list[self.task_list!=0]))
            index1 = route[-1].item()
            index2 = route[-2].item()

            current_position = self.target_set[:, index2]
            target_position = self.target_set[:, index1]

            direction = target_position - current_position
            self.angle = torch.atan2(direction[0][1], direction[0][0]).item()

            self.next_action_gap = (current_position - target_position).norm(p=2, dim=1).item()/self.velocity

            self.vehicle_state = target_position.unsqueeze(0)
        else:
            # if select 0, stay at previous state
            self.next_action_gap = 0

if __name__ == "__main__":
    from utils import load_train_config
    from env.static_env import TA_Static

    cfg = load_train_config("simple")

    env = TA_Static(cfg)
    env.reset()

    for v_type_name, v_info in env.vehicle_initial.items():
        for id in range(v_info["num"]):
            # TODO: Pilot_cfg should be re-developed
            v1 = Pilot(pilot_cfg=cfg, type_name=v_type_name, type_info=v_info, id=id)
            break
        break

    checkpoint = torch.load('model_save/simple/exp_1.pth')
    v1.load_state_dict(checkpoint['model'])

    print("TEST PASS!")
