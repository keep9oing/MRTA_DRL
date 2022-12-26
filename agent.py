import torch
import copy


class Agent():
    def __init__(self, agentID,
                 local_agent_encoder,
                 local_decoder,
                 local_target_encoder,
                 vehicle_type,
                 vehicle_pos,
                 vehicle_vel,
                 decode_type='sampling',
                 device=None):
        self.ID = agentID
        self.target_encoder = local_target_encoder
        self.agent_encoder = local_agent_encoder
        self.decoder = local_decoder
        self.target_encoder.share_memory()
        self.agent_encoder.share_memory()
        self.decoder.share_memory()
        self.device = device

        self.vehicle_type = vehicle_type
        self.initial_vehicle_state = torch.from_numpy(vehicle_pos).to(self.device)
        self.vehicle_pos = torch.from_numpy(vehicle_pos).to(self.device) # 1 x 1 x 2
        self.velocity = vehicle_vel

        self.task_list = torch.tensor([0], device=self.device)  # list to store task, start at initial pos
        self.action_list = []
        self.observation_agent = []
        self.observation_idle_embeddings = []
        self.observation_task = []
        self.observation_mask = []

        self.next_action_gap = 0  # time to select next target
        self.cost = 0
        self.finish = False  # finish flag
        self.decode_type = decode_type

    def get_vehicle_pos(self):
        if self.device == "cpu":
            # self.vehicle_state = torch.cat((self.vehicle_pos,torch.tensor([self.velocity])[None,None,:]), dim=2)
            return self.vehicle_pos.numpy()
        else:
            # self.vehicle_state=torch.cat((self.vehicle_pos,torch.tensor([self.velocity])[None,None,:]), dim=2)
            return self.vehicle_pos.cpu().numpy()

    def calculate_encoded_agent(self, agent_inputs):
        # print("pos:",self.vehicle_pos.shape)
        # print("inputs:",agent_inputs.shape)
        agent_inputs[0][:,:3] = agent_inputs[0][:,:3] - torch.cat((self.vehicle_pos,torch.tensor([[[0]]], device=self.device)),dim=-1)
        agent_feature = self.agent_encoder(agent_inputs)
        return agent_feature, agent_inputs

    def calculate_encoded_target(self, task_inputs):
        task_set = copy.deepcopy(task_inputs)
        task_inputs = task_set - self.vehicle_pos
        task_feature, idle_embedding = self.target_encoder(task_inputs)
        return task_feature, task_inputs, idle_embedding

    def select_next_target(self, obs):
        if 0 in obs["mask"][0,:]:

            agent_feature, agent_input = self.calculate_encoded_agent(agent_inputs=obs["fleet"])
            task_feature, task_input, idle_embedding = self.calculate_encoded_target(task_inputs=obs["task"])

            self.observation_agent.append(agent_input)
            self.observation_idle_embeddings.append(idle_embedding)
            self.observation_task.append(task_input)
            mask = copy.deepcopy(obs["mask"])
            mask = torch.cat((torch.tensor([[0]], device=self.device),mask), dim=-1)
            self.observation_mask.append(mask)

            next_target_index, log_prob = self.decoder(target_feature=task_feature,
                                                       current_state=torch.mean(task_feature,dim=1).unsqueeze(1),
                                                       agent_feature=agent_feature,
                                                       mask=mask,
                                                       decode_type=self.decode_type)
            self.action_list.append(next_target_index)
            self.task_list = torch.cat((self.task_list, next_target_index))
        else:
            self.finish = True
            next_target_index = None
        return next_target_index, self.finish

    def update_next_action_gap(self):
        if self.task_list[-1] != 0:
            route = torch.cat((self.task_list[None,0],self.task_list[self.task_list!=0]))
            index1 = route[-1].item()
            index2 = route[-2].item()
            current_position = self.target_set[:, index2]
            target_position = self.target_set[:, index1]
            self.next_action_gap = (current_position - target_position).norm(p=2, dim=1).item()/self.velocity
            self.vehicle_pos = target_position.unsqueeze(0)
        else:
            # if select 0, stay at previous state
            self.next_action_gap = 0

    def get_sum_flight_time(self):
        route = copy.deepcopy(self.task_list)
        route = torch.cat((route[None,0],route[route!=0]))

        d = torch.gather(input=self.target_set, dim=1, index=route[None, :, None].repeat(1, 1, 2))

        return (torch.sum((d[:, 1:] - d[:, :-1]).norm(p=2, dim=2), dim=1))/self.velocity

    def run(self, obs):
        ## TODO: various fleet types
        obs["fleet"] = torch.from_numpy(obs["fleet"]).to(self.device)
        ## TODO: various task types
        obs["task"] = torch.from_numpy(obs["task"]["visit"]["position"]).to(self.device) ## This implementation is temporary

        obs["mask"] = torch.from_numpy(obs["mask"]).to(self.device)

        self.target_set = torch.cat((self.initial_vehicle_state, obs["task"]), dim=1)

        next_target_index, finish = self.select_next_target(obs)

        if finish is not True:
            self.update_next_action_gap()
        else:
            self.next_action_gap = 0

        return next_target_index, finish
