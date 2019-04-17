import os
import sys
import time

import tensorflow as tf
import pandas as pd
import numpy as np

from enum import Enum
from utils.sth import sth
from utils.replay_buffer import ReplayBuffer
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import inspect_checkpoint as chkp
from mlagents.envs import UnityEnvironment

if sys.platform.startswith('win'):
    import win32api
    import win32con

EXIT = False

np.set_printoptions(threshold=10)
pd.set_option('max_colwidth', 0)
possible_output_nodes = ['out/Action']


class algorithms(Enum):
    ppo_sep_ac = 1  # AC, stochastic
    ppo_com = 2  # AC, stochastic
    # boundary, about the way of calculate `discounted reward`
    sac = 3  # AC+Q, stochastic, off-policy
    sac_no_v = 4
    ddpg = 5  # AC+Q, deterministic, off-policy
    td3 = 6  # AC+Q, deterministic, off-policy


train_config = {
    # choose algorithm
    'algorithm': algorithms.sac,
    'init_max_step': 300,
    # use for both on-policy and off-policy, control the max step within one episode.
    'max_step': 2500,
    'max_episode': 50000,
    'max_sample_time': 20,
    'till_all_done': True,  # use for on-policy leanring
    # train mode, .exe or unity-client && train or inference
    'train': True,
    'unity_mode': True,
    'unity_file': '',
    'port': 5008,
    # shuffle batch or not
    'random_batch': True,
    'batchsize': 100,
    'epoch': 10,
    # some sets about using replay_buffer
    'use_replay_buffer': True,  # on-policy or off-policy
    'use_priority': False,
    'buffer_size': 10000,
    'buffer_batch_size': 100,
    'max_learn_time': 20
}
hyper_config = {
    # set the temperature of SAC, auto adjust or not
    'alpha': 0.2,
    'auto_adaption': True,

    'ployak': 0.995,  # range from 0. to 1.
    'epsilon': 0.2,  # control the learning stepsize of clip-ppo
    'beta': 1.0e-3,  # coefficient of entropy regularizatione
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
    'max_episode': 50000,
    'base_sigma': 0.1,  # only work on stochastic policy
    'assign_interval': 4  # not use yet
}

def main():
    if sys.platform.startswith('win'):
        win32api.SetConsoleCtrlHandler(_win_handler, True)

    if train_config['unity_mode']:
        env = UnityEnvironment()
    else:
        env = UnityEnvironment(
            file_name=train_config['unity_file'],
            no_graphics=True if train_config['train'] else False,
            base_port=train_config['port']
        )
    brain_name = env.external_brain_names[0]
    brain = env.brains[brain_name]
    # set the memory use proportion of GPU
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default() as g:
        with tf.Session(graph=g, config=tf_config) as sess:
            print('Algorithm: {0}'.format(
                train_config['algorithm'].name))
            if train_config['algorithm'] == algorithms.ppo_sep_ac:
                from ppo.ppo_base import PPO_SEP
                model = PPO_SEP(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('PPO_SEP initialize success.')
            elif train_config['algorithm'] == algorithms.ppo_com:
                from ppo.ppo_base import PPO_COM
                model = PPO_COM(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('PPO_COM initialize success.')
            elif train_config['algorithm'] == algorithms.sac:
                from sac.sac import SAC
                model = SAC(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('SAC initialize success.')
            elif train_config['algorithm'] == algorithms.sac_no_v:
                from sac.sac_no_v import SAC_NO_V
                model = SAC_NO_V(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('SAC_NO_V initialize success.')
            elif train_config['algorithm'] == algorithms.ddpg:
                from ddpg.ddpg import DDPG
                model = DDPG(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('DDPG initialize success.')
            elif train_config['algorithm'] == algorithms.td3:
                from td3.td3 import TD3
                model = TD3(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                print('TD3 initialize success.')
            sess.run(tf.global_variables_initializer())
            try:
                if train_config['train']:
                    train_OffPolicy(
                        sess=sess,
                        env=env,
                        brain_name=brain_name,
                        begin_episode=0,
                        model=model,
                        hyper_config=hyper_config,
                        train_config=train_config
                    ) if not train_config['use_replay_buffer'] else train_OnPolicy(
                        sess=sess,
                        env=env,
                        brain_name=brain_name,
                        begin_episode=0,
                        model=model,
                        hyper_config=hyper_config,
                        train_config=train_config
                    )
                else:
                    inference(env, brain_name, model, train_config)
            except Exception as e:
                print(e)
            finally:
                env.close()
    sys.exit()


def inference(env, brain_name, model, train_config):
    sigma_offset = np.zeros(model.a_counts) + 0.0001
    while True:
        obs = env.reset(
            train_mode=train_config['train'])[brain_name]
        state = obs.vector_observations
        while True:
            _, action = model.choose_inference_action(
                s=state, sigma_offset=sigma_offset)
            obs = env.step(action)[brain_name]
            state = obs.vector_observations
            if EXIT:
                return


def train_OffPolicy(sess, env, brain_name, begin_episode, model, hyper_config, train_config):
    sigma_offset = np.zeros(model.a_counts) + hyper_config['base_sigma']
    for episode in range(begin_episode, train_config['max_episode']):
        print('-' * 30 + str(episode) + ' ๑乛◡乛๑ ' +
              train_config['algorithm'].name + '-' * 30)
        if EXIT:
            return
        step = 0
        total_reward = 0.
        total_discounted_reward = 0.
        discounted_reward = 0
        start = time.time()
        obs = env.reset(train_mode=True)[
            brain_name]
        agents_num = len(obs.agents)
        end = time.time()
        print(f'reset envs cost time: {end - start}')
        state_ = obs.vector_observations
        dones_flag = np.zeros(agents_num)
        start = time.time()
        data = {f'{i}': pd.DataFrame(columns=['state', 'action', 'old_prob', 'reward', 'next_state', 'done'])
                for i in range(agents_num)}
        end = time.time()
        print(f'create dataframe cost time: {end - start}')

        start = time.time()
        while True:
            state = state_
            prob, action = model.choose_action(
                s=state, sigma_offset=sigma_offset)
            obs = env.step(action)[brain_name]
            step += 1
            reward = obs.rewards
            state_ = obs.vector_observations
            dones_flag += obs.local_done
            for i in range(agents_num):
                data[f'{i}'] = data[f'{i}'].append({
                    'state': state[i],
                    'action': action[i],
                    'old_prob': prob[i] + 1e-10,
                    'next_state': state_[i],
                    'reward': reward[i],
                    'done': obs.local_done[i]
                }, ignore_index=True)
            if train_config['till_all_done']:
                sample_time = time.time() - start
                if all(dones_flag) or sample_time > train_config['max_sample_time']:
                    train_config['init_max_step'] = step
                    print(
                        f'(interactive)collect data cost time: {sample_time}')
                    break
            elif step >= train_config['init_max_step']:
                sample_time = time.time() - start
                print(
                    f'(interactive)collect data cost time: {sample_time}')
                if sample_time > train_config['max_sample_time']:
                    train_config['max_step'] = train_config['init_max_step']
                break
        start = time.time()

        dones = 0
        hits = 0
        for i in range(agents_num):
            done_index = data[f'{i}'][data[f'{i}'].done == True].index.tolist()
            hit_index = data[f'{i}'][data[f'{i}'].reward > 0].index.tolist()
            dones += len(done_index)
            hits += len(hit_index)
            if len(done_index):
                print(
                    f'[Agent {i}] dones: {len(done_index)} \thits: {len(hit_index)} \thit ratio: {len(hit_index)/len(done_index):.2%}')
            else:
                print(f'[Agent {i}] no done.')
            data[f'{i}']['value'] = model.get_state_value(
                s=data[f'{i}']['state'].values.tolist(), sigma_offset=sigma_offset)
            value_ = model.get_state_value(
                s=[state_[i]], sigma_offset=sigma_offset)
            if not data[f'{i}']['done'].values[-1]:
                discounted_reward = value_
            data[f'{i}']['total_reward'] = sth.discounted_sum(
                data[f'{i}']['reward'], 1, data[f'{i}']['reward'].values[-1], done_index, train_config['init_max_step'])
            if train_config['algorithm'].value <= 3:
                data[f'{i}']['discounted_reward'] = sth.discounted_sum(
                    data[f'{i}']['reward'], hyper_config['gamma'], discounted_reward, done_index, train_config['init_max_step'])
                data[f'{i}']['td_error'] = sth.discounted_sum_minus(
                    data[f'{i}']['reward'].values,
                    hyper_config['gamma'],
                    value_,
                    done_index,
                    data[f'{i}']['value'].values,
                    train_config['init_max_step']
                )
                data[f'{i}']['advantage'] = sth.discounted_sum(
                    data[f'{i}']['td_error'],
                    hyper_config['lambda'] * hyper_config['gamma'],
                    0,
                    done_index,
                    train_config['init_max_step']
                )
            else:
                data[f'{i}']['discounted_reward'] = sth.discounted_sum(
                    data[f'{i}']['reward'], hyper_config['gamma'], discounted_reward, done_index, train_config['init_max_step'], data[f'{i}']['value'])
                data[f'{i}']['advantage'] = None
            total_reward += (data[f'{i}']['total_reward'][0] / agents_num)
            total_discounted_reward += (data[f'{i}']
                                        ['discounted_reward'][0] / agents_num)
        if dones:
            print(
                f'#Agents Num#: {agents_num} \ttotal_dones: {dones} \ttotal_hits: {hits} \tratio: {hits/dones:.2%}')
        else:
            print(
                f'#Agents Num#: {agents_num} \tOMG! ALL AGENTS NO DONE.')
        end = time.time()
        print(f'calculate cost time: {end - start}')

        start = time.time()
        for j in range(agents_num):
            for _ in range(train_config['epoch']):
                for i in range(0, train_config['init_max_step'], train_config['batchsize']):
                    if train_config['random_batch']:
                        i_data = data[f'{j}'].sample(
                            n=train_config['batchsize']) if train_config['batchsize'] < train_config['init_max_step'] else data[f'{j}']
                    else:
                        i_data = data[f'{j}'].iloc[i:i
                                                   + train_config['batchsize'], :]
                    model.learn(
                        s=i_data['state'].values.tolist(),
                        a=i_data['action'].values.tolist(),
                        r=i_data['reward'].values[:, np.newaxis],
                        s_=i_data['next_state'].values.tolist(),
                        dc_r=i_data['discounted_reward'].values[:, np.newaxis],
                        episode=episode,
                        sigma_offset=sigma_offset,
                        old_prob=i_data['old_prob'].values.tolist(),
                        advantage=i_data['advantage'].values[:, np.newaxis]
                    )
        learn_time = time.time() - start
        print(f'learn cost time: {learn_time}')
        print('episede: {0} steps: {1} dc_reward: {2} reward: {3}'.format(
            episode, step, total_discounted_reward, total_reward))


def train_OnPolicy(sess, env, brain_name, begin_episode, model, hyper_config, train_config):
    sigma_offset = np.zeros(model.a_counts) + hyper_config['base_sigma']
    buffer = ReplayBuffer(model.s_dim, model.a_counts,
                          train_config['buffer_size'], train_config['use_priority'])
    for episode in range(begin_episode, train_config['max_episode']):
        print('-' * 30 + str(episode) + ' ๑乛◡乛๑ ' +
              train_config['algorithm'].name + '-' * 30)
        if EXIT:
            return
        step = 0
        obs = env.reset(train_mode=True)[
            brain_name]
        agents_num = len(obs.agents)
        total_reward = np.zeros(agents_num)
        total_discounted_reward = np.zeros(agents_num)
        state_ = obs.vector_observations
        hits_flag = np.zeros(agents_num, dtype=np.int32)
        dones_flag = np.zeros(agents_num, dtype=np.int32)
        start = time.time()
        while True:
            state = state_
            prob, action = model.choose_action(
                s=state, sigma_offset=sigma_offset)
            obs = env.step(action)[brain_name]
            step += 1
            reward = np.array(obs.rewards)
            for i in range(agents_num):
                if dones_flag[i] == 0:
                    total_reward[i] += reward[i]
                    total_discounted_reward[i] += hyper_config['gamma'] * reward[i]
            state_ = obs.vector_observations
            hits_flag += np.int64(reward > 0)
            dones_flag += obs.local_done
            dc_r = reward + hyper_config['gamma'] * model.get_state_value(
                s=state_, sigma_offset=sigma_offset)
            td_error = dc_r - model.get_state_value(
                s=state, sigma_offset=sigma_offset)
            advantage = np.zeros(agents_num)
            for i in range(agents_num):
                buffer.store(
                    state=state[i],
                    action=action[i],
                    prob=prob[i],
                    reward=reward[i],
                    discounted_reward=dc_r[i],
                    td_error=td_error[i],
                    next_state=state_[i],
                    advantage=advantage[i],
                    done=obs.local_done[i]
                )
            if buffer.buffer_size >= train_config['buffer_size']:
                data_from_buffer = buffer.sample_batch(
                    train_config['buffer_batch_size'])
                model.learn(
                    s=data_from_buffer['state'],
                    a=data_from_buffer['action'],
                    r=data_from_buffer['reward'][:, np.newaxis],
                    s_=data_from_buffer['next_state'],
                    dc_r=data_from_buffer['discounted_reward'][:, np.newaxis],
                    episode=episode,
                    sigma_offset=sigma_offset,
                    old_prob=data_from_buffer['old_prob'],
                    advantage=data_from_buffer['advantage'][:, np.newaxis]
                )
            if all(dones_flag) or step >= train_config['max_step']:
                break
        learn_time = time.time() - start
        print('learn_time: {0}'.format(
            learn_time))

        dones, hits = np.sum(dones_flag), np.sum(hits_flag)
        if dones:
            print(
                f'#Agents Num#: {agents_num} \ttotal_dones: {dones} \ttotal_hits: {hits} \tratio: {hits/dones:.2%}')
        else:
            print(
                f'#Agents Num#: {agents_num} \tOMG! ALL AGENTS NO DONE.')
        print('episede: {0} steps: {1} dc_reward: {2} reward: {3}\n'.format(
            episode, step, total_discounted_reward.mean(), total_reward.mean()))


def _win_handler(event):
    """
    This function gets triggered after ctrl-c or ctrl-break is pressed
    under Windows platform.
    """
    if event in (win32con.CTRL_C_EVENT, win32con.CTRL_BREAK_EVENT):
        global EXIT
        EXIT = True
        return True
    return False


if __name__ == '__main__':
    main()
