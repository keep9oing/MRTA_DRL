import numpy as np

v2type = {"uav_1": 0, "uav_2": 1}

class TA_Static():
  """
  static task allocation environment
  """
  def __init__(self, cfg, seed_ext=None):

    self.cfg = cfg
    self.device = cfg["device"]

    if seed_ext is not None:
      self.seed = seed_ext
      np.random.seed(self.seed)
    else:
      np.random.seed(None)

    self.task_config_initialize()
    self.vehicle_config_initialize()

  def task_config_initialize(self):

    self.task = dict()

    self.task_total_num = np.random.randint(self.cfg["task_config"]["num"][0],
                                            self.cfg["task_config"]["num"][1]+1)
    self.task_type = self.cfg["task_config"]["type"]

    t_num = 0
    t_temp = 0
    for t_idx, t_name in enumerate(self.task_type):
      self.task[t_name] = dict()

      if t_idx+1 < len(self.task_type):
        t_temp = np.random.randint(0, self.task_total_num+1-t_num)
      else:
        t_temp = self.task_total_num-t_num

      self.task[t_name]["num"] = t_temp
      t_num += t_temp

    assert t_num == self.task_total_num, "t_num:{}, task_total_num:{}".format(t_num, self.task_total_num)

  def vehicle_config_initialize(self):

    self.vehicle_initial = dict()
    self.vehicle_total_num = np.random.randint(self.cfg["vehicle_config"]["num"][0],
                                               self.cfg["vehicle_config"]["num"][1]+1)
    self.vehicle_type = self.cfg["vehicle_config"]["type"]

    v_num = 0
    v_temp = 0
    for v_idx, v_type in enumerate(self.vehicle_type):
      self.vehicle_initial[v_type] = dict()

      if v_idx+1 < len(self.vehicle_type):
        v_temp = np.random.randint(0, self.vehicle_total_num+1-v_num)
      else:
        v_temp = self.vehicle_total_num - v_num

      self.vehicle_initial[v_type]["num"] = v_temp
      self.vehicle_initial[v_type]["velocity"] = self.cfg["vehicle_config"][v_type]["velocity"]
      self.vehicle_initial[v_type]["angle"] = 0
      self.vehicle_initial[v_type]["task"] = self.cfg["vehicle_config"][v_type]["task"]
      v_num += v_temp

    assert v_num == self.vehicle_total_num, "v_num:{}, vehicle_total_num:{}, info:{}".format(v_num, self.vehicle_total_num)

  def reset(self, task_ext=None, vehicle_ext=None):

    if task_ext is not None:
      self.task = task_ext # manual initializatoin
    else:
      self.generate_task_state()

    if vehicle_ext is not None:
      self.vehicle_initial = vehicle_ext # manual initializatoin
    else:
      self.generate_vehicle_initial_state()

    self.generate_mask()

    self.agent_action_gap_list = np.zeros(self.vehicle_total_num)
    self.waiting_agent_list = np.zeros(self.vehicle_total_num)

  def generate_task_state(self):
    for t_name, t_config in self.task.items():
      self.task[t_name]["position"] = np.random.uniform(low=0, high=1, size=(1, t_config["num"], 2)).astype(np.float32)

  def generate_vehicle_initial_state(self):
    for v_type, v_config in self.vehicle_initial.items():
      self.vehicle_initial[v_type]["position"] = np.random.uniform(low=0, high=1, size=(1, v_config["num"], 2)).astype(np.float32)

  def generate_mask(self):
    self.global_mask = np.zeros((1, self.task_total_num)).astype(np.int64)

  def update_mask(self, target_index):
    np.put_along_axis(self.global_mask, indices=np.array([[target_index-1]]), axis=1, values=1)

  def get_vehicle_observation(self, agentList, agent_id):
    """
        return: observation(dict)
                fleet: 1 x veh_num x (state_dim+gap_dim)
                task: 1 x task_num x task_dim
                mask: 1 x task_num
    """
    vehicle_pos = []
    vehicle_vel = []
    vehicle_next_action_gap = []
    vehicle_type = []
    vehicle_partial_task_list = []

    for i in range(self.vehicle_total_num):
        vehicle_pos.append(agentList[i].get_vehicle_pos())
        vehicle_vel.append(agentList[i].velocity)
        vehicle_next_action_gap.append(agentList[i].next_action_gap)
        vehicle_type.append(v2type[agentList[i].vehicle_type])
        vehicle_partial_task_list.append(agentList[i].task_list)

    vehicle_pos = np.stack(vehicle_pos, axis=2).squeeze(0) # 1 x veh_num x pos(2)
    vehicle_next_action_gap = np.array(vehicle_next_action_gap)[None,:,None] # 1 x gap_dim(1) x 1
    vehicle_vel = np.array(vehicle_vel)[None,:,None]# 1 x vel_dim(1) x 1

    vehicle_type = np.array(vehicle_type)[None,:,None]
    vehicle_type_onehot = np.zeros((1,vehicle_vel.shape[1],len(v2type))) # 1 x veh_num x type_dim(depend on the number of types)
    np.put_along_axis(vehicle_type_onehot, vehicle_type, 1, axis=2)

    vehicle_state_observation = np.concatenate((vehicle_pos, vehicle_next_action_gap, vehicle_vel, vehicle_type_onehot), axis=-1).astype(np.float32) # 1 x veh_hum x (pos+gap+vel)
    # vehicle_state_observation = np.concatenate((vehicle_pos, vehicle_next_action_gap, vehicle_vel), axis=-1).astype(np.float32) # 1 x veh_hum x (pos+gap+vel)

    observation = {"fleet": vehicle_state_observation, "task":self.task, "mask": self.global_mask}

    return observation

  def step(self, agent, action):
    # TODO: Not yet
    # TODO: Maybe referencing petting zoo style
    if agent.finish:
        self.waiting_agent_list[agent.ID] = 1

    if action == 0:
        self.waiting_agent_list[agent.ID] = 1

    if action is not None:
        if action.item() != 0:
            self.waiting_agent_list[agent.ID] = 0
            self.update_mask(action.item())

    self.agent_action_gap_list[agent.ID] = agent.next_action_gap
