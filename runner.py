import torch
import torch.nn as nn
from agent import Agent
from agentEncoder import AgentEncoder
from targetEncoder import TargetEncoder
from decoder import Decoder


class ModelRunner(nn.Module):
    def __init__(self, cfg, decode_type='sampling', training=True):
        super(ModelRunner, self).__init__()
        self.cfg = cfg

        self.local_agent_encoder = AgentEncoder(self.cfg)
        self.local_target_encoder = TargetEncoder(self.cfg)
        self.local_decoder = Decoder(self.cfg)
        self.decode_type = decode_type

        self.device = self.cfg["device"]

    def forward(self, env):
        vehicle_amount = env.vehicle_total_num

        agentList = []
        route_set = dict()
        agent_id = 0

        for v_type in env.vehicle_type:
            vehicle_state = env.vehicle_initial[v_type]
            for i in range(vehicle_state["num"]):
                route_set[agent_id]={"type":v_type}
                agentList.append(Agent(agentID=agent_id,
                                local_agent_encoder=self.local_agent_encoder,
                                local_target_encoder=self.local_target_encoder,
                                local_decoder=self.local_decoder,
                                vehicle_type = v_type,
                                vehicle_pos = vehicle_state["position"][0,i,None,None],
                                vehicle_vel = vehicle_state["velocity"],
                                decode_type=self.decode_type,
                                device = self.device))
                agent_id += 1

        # for v in self.cfg.vehicle_config:
        #     vehicle_state = env.vehicle_initial[v.type_name]
        #     for i in range(vehicle_state["num"]):
        #         agentList.append(Agent(agentID=i,
        #                         local_agent_encoder=self.local_agent_encoder,
        #                         local_target_encoder=self.local_target_encoder,
        #                         local_decoder=self.local_decoder,
        #                         vehicle_pos = vehicle_state["position"][0,i,None,None],
        #                         vehicle_vel = vehicle_state["velocity"],
        #                         decode_type=self.decode_type,
        #                         device = self.device))

        agent_action_gap_list = torch.zeros(vehicle_amount)
        waiting_agent_list = torch.zeros(vehicle_amount)

        while True:
            all_finished = True

            for i in range(vehicle_amount):
                if agentList[i].next_action_gap <= torch.finfo(torch.float32).eps:
                    obs = env.get_vehicle_observation(agentList,i)
                    next_target_index, _ = agentList[i].run(obs)
                    # env.step(agentList[i], next_target_index)

                    if agentList[i].finish:
                        waiting_agent_list[i] = 1

                    if next_target_index == 0:
                        waiting_agent_list[i] = 1

                    if next_target_index is not None:
                        if next_target_index.item() != 0:
                            waiting_agent_list[i] = 0
                            env.update_mask(next_target_index.item())

                    agent_action_gap_list[i] = agentList[i].next_action_gap

                all_finished = all_finished and agentList[i].finish

            # Cost of flight time.
            if all_finished:
                for i in range(vehicle_amount):
                    agentList[i].cost += agentList[i].get_sum_flight_time()
                break

            # Get penalty when every agent is idle before finishing whole mission.
            # if int(env.waiting_agent_list.sum())==vehicle_amount:
            #     for i in range(vehicle_amount):
            #         agentList[i].cost += agentList[i].get_sum_flight_time()+10
            #     break
            if int(waiting_agent_list.sum())==vehicle_amount:
                for i in range(vehicle_amount):
                    agentList[i].cost += agentList[i].get_sum_flight_time()+100
                break

            # if env.waiting_agent_list.sum() < vehicle_amount:
            #     minimum_select_gap = env.agent_action_gap_list[env.waiting_agent_list<1].min()
            if waiting_agent_list.sum() < vehicle_amount:
                minimum_select_gap = agent_action_gap_list[waiting_agent_list<1].min()

            # for i in range(vehicle_amount):
            #     agentList[i].next_action_gap -= minimum_select_gap
            #     env.agent_action_gap_list[i] = agentList[i].next_action_gap
            #     if agentList[i].next_action_gap < 0:
            #         agentList[i].next_action_gap = 0

            for i in range(vehicle_amount):
                agentList[i].next_action_gap -= minimum_select_gap
                agent_action_gap_list[i] = agentList[i].next_action_gap
                if agentList[i].next_action_gap < 0:
                    agentList[i].next_action_gap = 0

        flight_time_list = []
        reward_list = []

        episode_buffer = [[] for _ in range(5)]

        for i in range(vehicle_amount):
            flight_time_list.append(agentList[i].get_sum_flight_time())
            reward_list.append(agentList[i].cost)
            route_set[i]["route"] = agentList[i].task_list.tolist()
            episode_buffer[0] += agentList[i].observation_agent
            episode_buffer[1] += agentList[i].observation_idle_embeddings
            episode_buffer[2] += agentList[i].observation_task
            episode_buffer[3] += agentList[i].observation_mask
            episode_buffer[4] += agentList[i].action_list

        flight_time_list = torch.stack(flight_time_list)  # [vehicle_amount,1]
        reward_list = torch.stack(reward_list)

        per_agent_reward = -torch.max(reward_list).unsqueeze(0).repeat(len(episode_buffer[4]))
        total_reward = -torch.max(reward_list)
        max_flight_time, max_id = torch.max(flight_time_list,0)

        return route_set, per_agent_reward, total_reward, -max_flight_time, max_flight_time, max_id.item(), episode_buffer
