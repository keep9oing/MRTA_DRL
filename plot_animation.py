import numpy as np
import random
import os

import imageio
import matplotlib.pyplot as plt

import torch

from utils import load_model_config
from env import static_env
from pilot_module import Pilot

import numpy as np

save_gif = False

config_name = "simple_het"
exp_num = 'server_1'
cfg = load_model_config(config_name, exp_num)
cfg["device"] = "cpu"

checkpoint = torch.load('model_save/{}/exp_{}.pth'.format(config_name, exp_num))

fig, ax = plt.subplots()

while True:
    seed = random.randint(0, 1000)
    env = static_env.TA_Static(cfg, seed)
    env.reset()

    # Unpack information
    vehicle_info = env.vehicle_initial
    task_info = env.task
    vehicle_amount = env.vehicle_total_num
    task_amount = env.task_total_num

    # Prepare Vehicles
    vehicleList = []
    for v_type_name, v_info in vehicle_info.items():
        for id in range(v_info["num"]):
            vehicleList.append(Pilot(pilot_cfg=cfg, type_name=v_type_name, type_info=v_info, id=id))

    for i in range(vehicle_amount):
        vehicleList[i].to(cfg["device"])
        vehicleList[i].load_state_dict(checkpoint['model'])


    # Set Plot
    ax.set_title(config_name)
    ax.set_xlim((-0.1,1.1))
    ax.set_ylim((-0.1,1.1))
    ax.set_aspect("equal")

    ## MANUAL Color
    task_colors = ["k","m","c"]
    for i, (t_type_name, t_info) in enumerate(task_info.items()):
        ax.scatter(task_info[t_type_name]["position"][0][:,0],task_info[t_type_name]["position"][0][:,1],
                   marker='x', s=25,  facecolor=task_colors[i], label=t_type_name)


    colors = iter([plt.cm.Set1(i) for i in range(vehicle_amount+1)])
    vehicle_color = dict()
    vehicle_scatters = []
    vehicle_markers = ['^','s','o']
    for i, (v_type_name, v_info) in enumerate(vehicle_info.items()):
        vehicle_color[v_type_name]=next(colors)
        vehicle_depot = v_info["position"][0]

        ax.scatter(vehicle_depot[:,0], vehicle_depot[:,1], marker='*', s=80, facecolor=vehicle_color[v_type_name])
        for j in range(v_info["num"]):
            if j == 0:
                vehicle_scatters.append(ax.scatter(vehicle_depot[j,0], vehicle_depot[j,1],
                                                marker=vehicle_markers[i],s=80, facecolors=vehicle_color[v_type_name], label=v_type_name))
            else:
                vehicle_scatters.append(ax.scatter(vehicle_depot[j,0], vehicle_depot[j,1],
                                                marker=vehicle_markers[i],s=80, facecolors=vehicle_color[v_type_name]))

    ax.legend()

    # Simulation parameters
    global_step = 0

    vehicle_pos = []
    for v in vehicleList:
        vehicle_pos.append(list(v.get_vehicle_pos()[0,0]))
    vehicle_pos = np.array(vehicle_pos)
    vehicle_angle = np.zeros(vehicle_amount)
    vehicle_gap = np.zeros(vehicle_amount)
    vehicle_target = np.zeros(vehicle_amount)

    finish = False
    time_step = 0.005

    filenames = []
    if save_gif:
        if not os.path.exists("gif"):
            os.makedirs("gif")

    waiting_veh_list = np.zeros(vehicle_amount)
    while not finish:
        for v in range(vehicle_amount):
            vehicle = vehicleList[v]
            if vehicle.next_action_gap <= 0 and not env.global_mask[0].all() and waiting_veh_list[v] < 1:
                vehicle_pos[v] = vehicle.get_vehicle_pos()[0]

                local_obs = env.get_vehicle_observation(vehicleList, v)

                next_task, _ = vehicle(local_obs)

                print(f"VEHICLE {v} to Task {next_task}")
                if next_task == 0:
                    print(f"VEHICLE {v} IDLE")
                    waiting_veh_list[v] = 1

                vehicle_target[v] = next_task.item()
                vehicle_gap[v] = vehicle.next_action_gap
                vehicle_angle[v] = vehicle.angle
                vehicle_route = vehicle.get_route()
                vehicle_target_set = vehicle.get_target_set()[0]
                if len(vehicle_route) >= 2:
                    route = [vehicle_route[-2], vehicle_route[-1]]

                if next_task is not None:
                    if next_task.item() != 0:
                        env.update_mask(next_task.item())

                        ax.plot(vehicle_target_set[route][:,0],vehicle_target_set[route][:,1],
                        color=vehicle_color[vehicle.vehicle_type],
                        linestyle=':')

        # Step Simulation
        global_step += time_step

        active_vehicle_id = np.where(np.all([vehicle_gap > 0,vehicle_target!=0],axis=0))[0]
        type_idx = 0
        v_x = np.array([])
        v_y = np.array([])
        for v_type, v_info in vehicle_info.items():
            v_x = np.concatenate((v_x,v_info["velocity"] * np.cos(vehicle_angle[type_idx:type_idx+v_info["num"]])))
            v_y = np.concatenate((v_y,v_info["velocity"] * np.sin(vehicle_angle[type_idx:type_idx+v_info["num"]])))
            type_idx += v_info["num"]

        vehicle_pos[active_vehicle_id,0] += v_x[active_vehicle_id] * time_step
        vehicle_pos[active_vehicle_id,1] += v_y[active_vehicle_id] * time_step

        vehicle_gap -= time_step

        ## Vehicle Plot
        for v in range(vehicle_amount):
            if v in active_vehicle_id:
                vehicle_scatters[v].set_offsets(np.c_[vehicle_pos[v,0], vehicle_pos[v,1]])
                vehicleList[v].next_action_gap -= time_step

        if env.global_mask[0].all() and (vehicle_gap<=0).all():
            finish = True
        else:
            plt.pause(0.01)
            if save_gif:
                filename = f'./gif/{global_step}.png'
                filenames.append(filename)
                plt.savefig(filename)
    plt.cla()

    if save_gif:
        #build gif
        files=[]
        for filename in filenames:
            image = imageio.imread(filename)
            files.append(image)
        imageio.mimsave("./gif/mygif.gif", files, format='GIF', fps = 30)
        with imageio.get_writer('./gif/mygif.gif', mode='I') as writer:
            for filename in filenames:
                image = imageio.imread(filename)
                writer.append_data(image)

        # Remove files
        for filename in set(filenames):
            os.remove(filename)
        break
