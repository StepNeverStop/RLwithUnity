import os
import platform
from enum import Enum

# judge the current operating system
base = r'C:' if platform.system() == "Windows" or platform.system() == "Darwin" else r'/data'

env_list = [
    'RollerBall',
    '3DBall',
    'Boat'
]

reset_config = [None, {
    'copy': 10
}]

class algorithms(Enum):
    ppo_sep_ac = 1  # AC, stochastic
    ppo_com = 2  # AC, stochastic
    # boundary, about the way of calculate `discounted reward`
    sac = 3  # AC+Q, stochastic
    sac_no_v = 4
    ddpg = 5  # AC+Q, deterministic
    td3 = 6  # AC+Q, deterministic


unity_file = [
    r'C:/UnityBuild/RollerBall/custom/RollerBall-custom.exe',
    r'3dball',
    r'C:/UnityBuild/Boat/first/BoatTrain.exe',
    r'C:/UnityBuild/Boat/second/BoatTrain.exe',
    r'C:/UnityBuild/Boat/third/BoatTrain.exe',
    r'C:/UnityBuild/Boat/addTimePenalty/BoatTrain.exe'
]

max_episode = 5.0e4  # max episode num or step num, depend on whether episode-update or step-update

config = {
    'hyper parameters': {
        # set the temperature of SAC, auto adjust or not
        'alpha': 0.2,
        'auto_adaption': True,

        'ployak': 0.995, # range from 0. to 1.
        'epsilon': 0.2, # control the learning stepsize of clip-ppo 
        'beta': 1.0e-3, # coefficient of entropy regularizatione
        'lr': 5.0e-4,
        'actor_lr': 0.0001,
        'critic_lr': 0.0002,
        'tp_lr': 0.001,
        'reward_lr': 0.001,
        'gamma': 0.99,
        'lambda': 0.95,
        'action_bound': 1,
        'decay_rate': 0.7,
        'decay_steps': 100,
        'stair': False,
        'max_episode': max_episode,
        'base_sigma': 0.1, # only work on stochastic policy
        'assign_interval': 4 # not use yet
    },

    'train config': {
        # choose algorithm
        'algorithm': algorithms.sac_no_v,
        'init_max_step': 300,
        'max_step': 2500,
        'max_episode': max_episode,
        'max_sample_time': 5,
        'till_all_done': True,
        'start_continuous_done': False,
        # train mode, .exe or unity-client && train or inference
        'train': True,
        'unity_mode': False,
        'unity_file': unity_file[0].replace('C:',f'{base}'),
        'port': 5006,
        # trick
        'use_trick': False,
        # excel
        'excel_record': False,
        'excel_record_frequency': 10,
        # mongodb
        'mongo_record': False,
        'mongo_record_frequency': 10,
        'mongo_record_all': False,
        # shuffle batch or not
        'random_batch': True,
        'batchsize': 100,
        'epoch': 10,
        # checkpoint
        'save_frequency': 20,
        # set the agents' number and control mode
        'dynamic_allocation': True,
        'reset_config': reset_config[1],
        # deprecated and not recommend, 'cause increase the agents number will conflict with the sample steps under the limit of fixed max_sample_time
        'max_learn_time': None,
        # some sets about using replay_buffer
        'use_replay_buffer': False,
        'buffer_size' : 1.0e4,
        'buffer_batch_size': 100
    },

    'record config': {
        'basic_dir': r'C:/RLData/'.replace('C:',f'{base}'),
        'log_basic_dir': r'C:/RLData/logs/'.replace('C:',f'{base}'),
        'excel_basic_dir': r'C:/RLData/excels/'.replace('C:',f'{base}'),
        'checkpoint_basic_dir': r'C:/RLData/models/'.replace('C:',f'{base}'),
        'config_basic_dir': r'C:/RLData/config/'.replace('C:',f'{base}'),
        'project_name': env_list[0],
        'remark': r'testSAV_NO_V',
        'run_id': r'3',
        'logger2file' : False
    },

    'config_file': r"",
    'ps': r"",

    'clean_list': [

    ]
}
