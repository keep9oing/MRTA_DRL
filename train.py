import ray
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import os
import copy
import numpy as np
import time
from scipy.stats import ttest_rel

from runner import ModelRunner
from worker import Worker
from worker_for_test import TestWorker
from utils import load_train_config, load_ray_config, save_configs


config_name = "simple_het"

cfg = load_train_config(config_name)
exp_number = cfg["exp"]

ray_cfg = load_ray_config()

# Save Path
model_dir = "model_save/"+cfg["name"]
model_path = model_dir+"/exp_{}".format(exp_number)
if not os.path.exists(model_dir):
  os.makedirs(model_dir)
save_configs(config_name, model_path)

log_path = "log/"+cfg["name"]+"/exp_{}".format(exp_number)
writer = SummaryWriter(log_path)

np.random.seed(cfg["seed"])

ray.init()

device = cfg["device"]
global_model = ModelRunner(cfg)
global_model.share_memory()
global_model.to(device)

optimizer = optim.AdamW(global_model.parameters(), lr=cfg["lr"])

lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_decay_step"], gamma=cfg["lr_decay"])

worker_list = [Worker.remote(workerID=i, cfg=cfg) for i in range(ray_cfg["num_worker"])]

# info for tensorboard
average_loss = 0
average_advantage = 0
average_grad_norm = 0
average_rewards = 0
average_max_flight_time = 0
average_entropy = 0

global_step = 0

max_valid_value = -np.inf

baseline_value = None
test_set_num = ray_cfg["test_set_num"]
test_set = np.random.randint(low=0, high=1e8, size=[test_set_num // ray_cfg["num_test_worker"], ray_cfg["num_test_worker"]])

valid_set = np.random.randint(low=0, high=1e8, size=[test_set_num // ray_cfg["num_test_worker"], ray_cfg["num_test_worker"]])

if cfg["load_model"]:
    checkpoint = torch.load(model_path+".pth")
    global_step = checkpoint['step']
    # max_valid_value = checkpoint['valid_reward']
    # valid_set = checkpoint['valid_set']
    global_model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_decay.load_state_dict(checkpoint['lr_decay'])

    print("load model at", global_step)
    print(optimizer.state_dict()['param_groups'][0]['lr'])

# get global network weights
global_weights = global_model.state_dict()

# update local network
update_local_network_job_list = []
for i, worker in enumerate(worker_list):
    update_local_network_job_list.append(worker.set_model_weights.remote(global_weights))
baseline_weights = copy.deepcopy(global_weights)
update_baseline_network_job_list = []
for i, worker in enumerate(worker_list):
    update_baseline_network_job_list.append(worker.set_baseline_model_weights.remote(baseline_weights))

try:
    while True:
        global_step += 1

        sample_job_list = []
        for i, worker in enumerate(worker_list):
            sample_job_list.append(worker.sample.remote())

        if global_step % ray_cfg["epi_per_worker"] == 0:

            # get gradient and loss from runner
            get_gradient_job_list = []
            for i, worker in enumerate(worker_list):
                get_gradient_job_list.append(worker.return_gradient.remote())
            gradient_set_id, _ = ray.wait(get_gradient_job_list, num_returns=ray_cfg["num_worker"])
            gradient_loss_set = ray.get(gradient_set_id)

            for gradients, loss, grad_norm, advantage, max_flight_time,entropy, reward in gradient_loss_set:
                average_max_flight_time += max_flight_time
                average_loss += loss
                average_advantage += advantage
                average_grad_norm += grad_norm
                average_entropy += entropy
                average_rewards += reward

                optimizer.zero_grad()
                for g, global_param in zip(gradients, global_model.parameters()):
                    global_param._grad = g

                # update networks
                optimizer.step()

            if cfg["lr_decay"] < 1:
                lr_decay.step()

            update_local_network_job_list = []
            for i, worker in enumerate(worker_list):
                update_local_network_job_list.append(worker.set_model_weights.remote(global_weights))

        # tensorboard update
        if global_step % cfg["tensorboard_batch"] == 0:

            writer.add_scalar('main/reward', average_rewards / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('main/max_flight_time', average_max_flight_time / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('sub/loss',
                              average_loss / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('sub/entropy',
                              average_entropy / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('sub/advantage',
                              average_advantage / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('sub/grad_norm',
                              average_grad_norm / (ray_cfg["num_worker"] * cfg["tensorboard_batch"] / ray_cfg["epi_per_worker"]),
                              global_step)

            writer.add_scalar('etc/learning_rate',
                                optimizer.state_dict()['param_groups'][0]['lr'],
                                global_step)

            writer.add_scalar('etc/episode',
                              ray_cfg["num_worker"] * ray_cfg["epi_per_worker"] * global_step,
                              global_step)

            average_entropy = 0
            average_advantage = 0
            average_loss = 0
            average_grad_norm = 0
            average_rewards = 0
            average_max_flight_time = 0

        # update baseline model every 2048 steps
        if global_step % (cfg["update_baseline"]) == 0:
            # stop the training
            ray.wait(update_local_network_job_list, num_returns=ray_cfg["num_worker"])
            for a in worker_list:
                ray.kill(a)
            torch.cuda.empty_cache()
            time.sleep(5)
            print('evaluate baseline model at ', global_step)

            # test the baseline model on the new test set
            if baseline_value is None:
                test_worker_list = [TestWorker.remote(workerID=i, cfg=cfg, decode_type='greedy') for i in
                                    range(ray_cfg["num_test_worker"])]

                update_local_network_job_list = []

                for _, test_worker in enumerate(test_worker_list):
                    update_local_network_job_list.append(test_worker.set_weights.remote(baseline_weights))

                baseline_value = []
                for i in range(test_set_num // ray_cfg["num_test_worker"]):

                    sample_job_list = []

                    for j, test_worker in enumerate(test_worker_list):
                        sample_job_list.append(test_worker.sample.remote(cfg, test_set[i][j]))

                    sample_done_id, _ = ray.wait(sample_job_list, num_returns=ray_cfg["num_test_worker"])
                    results = ray.get(sample_done_id)
                    for reward, _ in results:
                        baseline_value.append(reward.item())

                for a in test_worker_list:
                    ray.kill(a)

            # test the current model's performance
            test_worker_list = [TestWorker.remote(workerID=i, cfg=cfg, decode_type='greedy') for i in
                                range(ray_cfg["num_test_worker"])]
            update_local_network_job_list = []
            for _, test_worker in enumerate(test_worker_list):
                update_local_network_job_list.append(test_worker.set_weights.remote(global_weights))

            test_value = []
            for i in range(test_set_num // ray_cfg["num_test_worker"]):
                sample_job_list = []
                for j, test_worker in enumerate(test_worker_list):
                    sample_job_list.append(test_worker.sample.remote(cfg, test_set[i][j]))
                sample_done_id, _ = ray.wait(sample_job_list, num_returns=ray_cfg["num_test_worker"])
                results = ray.get(sample_done_id)
                for reward, _ in results:
                    test_value.append(reward.item())

            for a in test_worker_list:
                ray.kill(a)

            time.sleep(5)

            # test the current model's performance to validation set
            valid_worker_list = [TestWorker.remote(workerID=i, cfg=cfg, decode_type='greedy') for i in
                                range(ray_cfg["num_test_worker"])]
            update_local_network_job_list = []
            for _, valid_worker in enumerate(valid_worker_list):
                update_local_network_job_list.append(valid_worker.set_weights.remote(global_weights))

            valid_value = []
            valid_max_flight_time = []
            for i in range(test_set_num // ray_cfg["num_test_worker"]):
                sample_job_list = []
                for j, valid_worker in enumerate(valid_worker_list):
                    sample_job_list.append(valid_worker.sample.remote(cfg, valid_set[i][j]))
                sample_done_id, _ = ray.wait(sample_job_list, num_returns=ray_cfg["num_test_worker"])
                results = ray.get(sample_done_id)
                for reward, max_flight_time in results:
                    valid_value.append(reward.item())
                    valid_max_flight_time.append(max_flight_time[0].item())
            valid_value = sum(valid_value)/len(valid_value)
            valid_max_time = sum(valid_max_flight_time)/len(valid_max_flight_time)

            writer.add_scalar('main/valid_reward',
                              valid_value,
                              global_step)
            writer.add_scalar('main/valid_max_flight_time',
                              valid_max_time,
                              global_step)

            for a in valid_worker_list:
                ray.kill(a)

            time.sleep(5)

            # restart training
            print('lr', optimizer.state_dict()['param_groups'][0]['lr'])
            worker_list = [Worker.remote(workerID=i, cfg=cfg) for i in range(ray_cfg["num_worker"])]

            for i, worker in enumerate(worker_list):
                update_local_network_job_list.append(worker.set_model_weights.remote(global_weights))
            update_baseline_network_job_list = []
            for i, worker in enumerate(worker_list):
                update_baseline_network_job_list.append(worker.set_baseline_model_weights.remote(baseline_weights))

            # update baseline if the model improved more than 5%
            test_avg_reward = sum(test_value)/len(test_value)
            baseline_avg_reward = sum(test_value)/len(test_value)
            print('test reward', test_avg_reward)
            print('baseline reward', baseline_avg_reward)
            if test_avg_reward > baseline_avg_reward:
                _, p = ttest_rel(test_value, baseline_value)
                print('p value', p)
                if p < 0.05:
                    print('update baseline model at ', global_step)
                    global_weights = global_model.state_dict()

                    baseline_weights = copy.deepcopy(global_weights)
                    update_baseline_network_job_list = []
                    for i, worker in enumerate(worker_list):
                        update_baseline_network_job_list.append(worker.set_baseline_model_weights.remote(baseline_weights))

                    test_set = np.random.randint(low=0, high=1e8,
                                                    size=[test_set_num // ray_cfg["num_test_worker"], ray_cfg["num_test_worker"]])
                    print('update test set')
                    baseline_value = None

            # save model if validation reward is better than the last best validation reward
            if valid_value > max_valid_value:
                print("GOOD! SAVE MODEL")
                max_valid_value = valid_value
                model_states = {"model": global_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_decay": lr_decay.state_dict(),
                                "step": global_step,
                                "valid_reward": max_valid_value}
                                # "valid_set": valid_set}
                torch.save(obj=model_states, f=model_path+".pth")


except KeyboardInterrupt:
    print("CTRL-C pressed. killing remote workers")
    for a in worker_list:
        ray.kill(a)
