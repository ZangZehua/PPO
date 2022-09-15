import torch
import os
import datetime


class Config:
    # Common config
    random_seed = 0  # set random seed if required (0 = no random seed)
    train_time = str(datetime.datetime.now().replace(microsecond=0).strftime("%Y-%m-%d-%H-%M-%S"))

    # CUDA config
    device = torch.device('cpu')
    device_name = "cpu"
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(device)

    # Env config
    env = "PongNoFrameskip-v4"
    worker_number = 8
    has_continuous_action_space = False
    action_dim = 6

    # Network config
    obs_channels = 4
    fc_in_dim = 7 * 7 * 64

    lr = 0.001
    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    # PPO train config
    max_step = 5000000
    update_timestep = 500 * 2  # update policy every n timesteps
    K_epochs = 20  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99
    lamb = 1
    c1 = 0.5
    c2 = 0.01

    # print config
    upgrade_freq = 500

    # save config
    if not os.path.exists("runs"):
        os.mkdir("runs")
    if not os.path.exists("runs/" + env_name + "_DQN"):
        os.mkdir("runs/" + env_name + "_DQN")
    reward_writer_path = "runs/" + env_name + "_DQN/" + train_time + "_train-reward"
    loss_writer_path = "runs/" + env_name + "_DQN/" + train_time + "_loss"

    if not os.path.exists("saved"):
        os.mkdir("saved")
    if not os.path.exists("saved/" + env_name + "_DQN"):
        os.mkdir("saved/" + env_name + "_DQN")
    if not os.path.exists("saved/" + env_name + "_DQN/models"):
        os.mkdir("saved/" + env_name + "_DQN/models")
    model_save_path = "saved/" + env_name + "_DQN/models/" + train_time + ".pth"


