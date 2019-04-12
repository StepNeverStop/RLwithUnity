import os
import sys
import json
import time
import logging

import tensorflow as tf
import pandas as pd
import numpy as np

import config_file
from utils.recorder import Recorder
from utils.sth import sth
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import inspect_checkpoint as chkp
from mlagents.envs import UnityEnvironment

if sys.platform.startswith('win'):
    import win32api
    import win32con
# from mlagents.trainers import tensorflow_to_barracuda as tf2bc  #since ml-agents v0.7.0

EXIT = False
# record_config['run_id'] = 'rollerAgentClassicRewardTest'+time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

np.set_printoptions(threshold=10)
# 保证所有数据能够显示，而不是用省略号表示，np.inf表示一个足够大的数
# np.set_printoptions(threshold = np.inf)
# 若想不以科学计数显示:
# np.set_printoptions(suppress = True)
pd.set_option('max_colwidth', 0)
possible_output_nodes = ['out/Action']

# print restore_op_name and filename_tensor_name
# print(saver.saver_def.filename_tensor_name)
# print(saver.saver_def.restore_op_name)


def main():
    if sys.platform.startswith('win'):
        # Add the _win_handler function to the windows console's handler function list
        win32api.SetConsoleCtrlHandler(_win_handler, True)
    if os.path.exists(os.path.join(config_file.config['config_file'], 'config.yaml')):
        config = sth.load_config(config_file.config['config_file'])
    else:
        config = config_file.config
        print(f'load config from config.')

    hyper_config = config['hyper parameters']
    train_config = config['train config']
    record_config = config['record config']

    basic_dir = record_config['basic_dir']
    last_name = record_config['project_name'] + '/' \
        + record_config['remark'] \
        + record_config['run_id']
    cp_dir = record_config['checkpoint_basic_dir'] + last_name
    cp_file = cp_dir + '/rb'
    log_dir = record_config['log_basic_dir'] + last_name
    excel_dir = record_config['excel_basic_dir'] + last_name
    config_dir = record_config['config_basic_dir'] + last_name
    sth.check_or_create(basic_dir, 'basic')
    sth.check_or_create(cp_dir, 'checkpoints')
    sth.check_or_create(log_dir, 'logs(summaries)')
    sth.check_or_create(excel_dir, 'excel')
    sth.check_or_create(config_dir, 'config')

    logger = create_logger(
        name='logger',
        console_level=logging.INFO,
        console_format='%(levelname)s : %(message)s',
        logger2file=record_config['logger2file'],
        file_name=log_dir + '\log.txt',
        file_level=logging.WARNING,
        file_format='%(lineno)d - %(asctime)s - %(module)s - %(funcName)s - %(levelname)s - %(message)s'
    )
    if train_config['train']:
        sth.save_config(config_dir, config)

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
            logger.info('Algorithm: {0}'.format(
                train_config['algorithm'].name))
            if train_config['algorithm'] == config_file.algorithms.ppo_sep_ac:
                from ppo.ppo_base import PPO_SEP
                model = PPO_SEP(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                logger.info('PPO_SEP initialize success.')
            elif train_config['algorithm'] == config_file.algorithms.ppo_com:
                from ppo.ppo_base import PPO_COM
                model = PPO_COM(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                logger.info('PPO_COM initialize success.')
            elif train_config['algorithm'] == config_file.algorithms.sac:
                from sac.sac import SAC
                model = SAC(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                logger.info('SAC initialize success.')
            elif train_config['algorithm'] == config_file.algorithms.ddpg:
                from ddpg.ddpg import DDPG
                model = DDPG(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                logger.info('DDPG initialize success.')
            elif train_config['algorithm'] == config_file.algorithms.td3:
                from td3.td3 import TD3
                model = TD3(
                    sess=sess,
                    s_dim=brain.vector_observation_space_size,
                    a_counts=brain.vector_action_space_size[0],
                    hyper_config=hyper_config
                )
                logger.info('TD3 initialize success.')
            recorder = Recorder(log_dir, excel_dir, record_config, logger,
                                max_to_keep=5, pad_step_number=True, graph=g)
            episode = init_or_restore(cp_dir, sess, recorder, cp_file)
            try:
                if train_config['train']:
                    train(
                        sess=sess,
                        env=env,
                        brain_name=brain_name,
                        begin_episode=episode,
                        model=model,
                        recorder=recorder,
                        cp_file=cp_file,
                        hyper_config=hyper_config,
                        train_config=train_config
                    )
                    tf.train.write_graph(
                        g, cp_dir, 'raw_graph_def.pb', as_text=False)
                    export_model(cp_dir, g)
                else:
                    inference(env, brain_name, model, train_config)
            except Exception as e:
                logger.error(e)
            finally:
                env.close()
    recorder.close()
    sys.exit()


def inference(env, brain_name, model, train_config):
    sigma_offset = np.zeros(model.a_counts) + 0.0001
    while True:
        obs = env.reset(
            config=train_config['reset_config'], train_mode=train_config['train'])[brain_name]
        state = obs.vector_observations
        while True:
            _, action = model.choose_inference_action(
                s=state, sigma_offset=sigma_offset)
            obs = env.step(action)[brain_name]
            state = obs.vector_observations
            if EXIT:
                return


def train(sess, env, brain_name, begin_episode, model, recorder, cp_file, hyper_config, train_config):
    base_agents_num = train_config['reset_config']['copy']
    sigma_offset = np.zeros(model.a_counts) + hyper_config['base_sigma']
    for episode in range(begin_episode, train_config['max_episode']):
        recorder.logger.info('-' * 30 + str(episode) + ' ๑乛◡乛๑ '
                             + train_config['algorithm'].name + '-' * 30)
        if EXIT:
            return
        if(episode % train_config['save_frequency'] == 0):
            start = time.time()
            recorder.saver.save(
                sess, cp_file, global_step=episode, write_meta_graph=False)
            end = time.time()
            recorder.logger.info(f'save checkpoint cost time: {end - start}')
        model_lr = model.decay_lr(episode)
        step = 0
        total_reward = 0.
        total_discounted_reward = 0.
        discounted_reward = 0
        flag = False
        # start = time.time()
        # for i in range(agents_num):
        #     data[f'{i}'].drop(data[f'{i}'].index, inplace=True)
        # end = time.time()
        # recorder.logger.info(f'clear dataframe cost time: {end - start}')
        start = time.time()
        obs = env.reset(config=train_config['reset_config'], train_mode=True)[
            brain_name]
        agents_num = len(obs.agents)
        end = time.time()
        recorder.logger.info(f'reset envs cost time: {end - start}')
        state_ = obs.vector_observations
        dones_flag = np.zeros(agents_num)
        dones_flag_sup = np.full(
            agents_num, -1) if train_config['start_continuous_done'] else np.zeros(agents_num)
        if not train_config['use_trick']:
            sigma_offset = np.zeros(model.a_counts) + \
                hyper_config['base_sigma']
        start = time.time()
        data = {f'{i}': pd.DataFrame(columns=['state', 'action', 'old_prob', 'reward', 'next_state', 'done'])
                for i in range(agents_num)}
        end = time.time()
        recorder.logger.info(f'create dataframe cost time: {end - start}')

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
            dones_flag_sup += obs.local_done
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
                if all(dones_flag) and all(dones_flag_sup) or sample_time > train_config['max_sample_time']:
                    train_config['init_max_step'] = step
                    recorder.logger.info(
                        f'(interactive)collect data cost time: {sample_time}')
                    break
            elif step >= train_config['init_max_step']:
                sample_time = time.time() - start
                recorder.logger.info(
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
                recorder.logger.info(
                    f'[Agent {i}] dones: {len(done_index)} \thits: {len(hit_index)} \thit ratio: {len(hit_index)/len(done_index):.2%}')
            else:
                recorder.logger.info(f'[Agent {i}] no done.')
                flag = True
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
            recorder.logger.info(
                f'#Agents Num#: {agents_num} \ttotal_dones: {dones} \ttotal_hits: {hits} \tratio: {hits/dones:.2%}')
        else:
            recorder.logger.info(
                f'#Agents Num#: {agents_num} \tOMG! ALL AGENTS NO DONE.')
        end = time.time()
        recorder.logger.info(f'calculate cost time: {end - start}')
        '''
        excel record
        '''
        if train_config['excel_record'] and episode % train_config['excel_record_frequency'] == 0:
            start = time.time()
            data['0'].to_excel(recorder.excel_writer,
                               sheet_name=f'{episode}', index=True)
            recorder.excel_writer.save()
            end = time.time()
            recorder.logger.info(
                f'save data to excel cost time: {end - start}')
        '''
        mongodb record
        '''
        if train_config['mongo_record'] and episode % train_config['mongo_record_frequency'] == 0:
            start = time.time()
            if train_config['mongo_record_all']:
                for i in range(agents_num):
                    recorder.mongodb[f'e{episode}a{i}'].insert(
                        json.loads(data[f'{i}'].T.to_json()).values())
            else:
                recorder.mongodb[f'e{episode}a'].insert(
                    json.loads(data['0'].T.to_json()).values())
            end = time.time()
            recorder.logger.info(
                f'save data to MongoDB cost time: {end - start}')

        start = time.time()
        for j in range(agents_num):
            for _ in range(train_config['epoch']):
                for i in range(0, train_config['init_max_step'], train_config['batchsize']):
                    if train_config['random_batch']:
                        i_data = data[f'{j}'].sample(
                            n=train_config['batchsize']) if train_config['batchsize'] < train_config['init_max_step'] else data[f'{j}']
                    else:
                        i_data = data[f'{j}'].iloc[i:i +
                                                   train_config['batchsize'], :]
                    model.learn(
                        s=i_data['state'].values.tolist(),
                        a=i_data['action'].values.tolist(),
                        r=i_data['reward'].values[:,np.newaxis],
                        s_=i_data['next_state'].values.tolist(),
                        dc_r=i_data['discounted_reward'].values[:,np.newaxis],
                        episode=episode,
                        sigma_offset=sigma_offset,
                        old_prob=i_data['old_prob'].values.tolist(),
                        advantage=i_data['advantage'].values[:, np.newaxis]
                    )
        learn_time = time.time() - start
        recorder.logger.info(f'learn cost time: {learn_time}')

        if train_config['dynamic_allocation']:
            # train_config['reset_config']['copy'] += 1 if learn_time < train_config['max_learn_time'] else -1
            # train_config['reset_config']['copy'] = 1 if train_config['reset_config']['copy'] == 0 else train_config['reset_config']['copy']
            train_config['reset_config']['copy'] += 1 if hits > (agents_num * 2 if train_config['start_continuous_done']
                                                                 else agents_num) else (-2 if train_config['reset_config']['copy'] > base_agents_num else 0)

        start = time.time()
        a_loss = np.array([model.get_actor_loss(
            s=data[f'{i}']['state'].values.tolist(),
            sigma_offset=sigma_offset,
            a=data[f'{i}']['action'].values.tolist(),
            old_prob=data[f'{i}']['old_prob'].values.tolist(),
            advantage=data[f'{i}']['advantage'].values[:, np.newaxis]
        ) for i in range(agents_num)]).mean()
        c_loss = np.array([model.get_critic_loss(
            s=data[f'{i}']['state'].values.tolist(),
            a=data[f'{i}']['action'].values.tolist(),
            r=i_data['reward'].values[:,np.newaxis],
            s_=i_data['next_state'].values.tolist(),
            dc_r=data[f'{i}']['discounted_reward'].values[:, np.newaxis],
            sigma_offset=sigma_offset
        ) for i in range(agents_num)]).mean()
        entropy = np.array([model.get_entropy(s=data[f'{i}']['state'].values.tolist(
        ), sigma_offset=sigma_offset) for i in range(agents_num)]).mean(axis=0)
        sigma = np.array([model.get_sigma(s=data[f'{i}']['state'].values.tolist(
        ), sigma_offset=sigma_offset) for i in range(agents_num)]).mean(axis=0)
        sigma_offset = np.array([np.log(c_loss + 1)]
                                * model.a_counts) + hyper_config['base_sigma']
        end = time.time()
        recorder.logger.info(f'get statistics cost time: {end - start}')

        writer_summary(recorder.writer, episode, [{
            'tag': 'TIME/sample_time',
            'value': sample_time
        },
            {
            'tag': 'TIME/steps',
            'value': step
        },
            {
            'tag': 'TIME/agents_num',
            'value': agents_num
        }])
        writer_summary(recorder.writer, episode, [{
            'tag': 'REWARD/discounted_reward',
            'value': total_discounted_reward
        },
            {
            'tag': 'REWARD/reward',
            'value': total_reward
        },
            {
            'tag': 'LEARNING_RATE/lr',
            'value': model_lr
        },
            {
            'tag': 'LOSS/actor_loss',
            'value': a_loss
        },
            {
            'tag': 'LOSS/critic_loss',
            'value': c_loss
        },
            {
            'tag': 'LOSS/actor_entropy_max',
            'value': entropy.max()
        },
            {
            'tag': 'LOSS/actor_entropy_min',
            'value': entropy.min()
        },
            {
            'tag': 'LOSS/actor_entropy_mean',
            'value': entropy.mean()
        },
            {
            'tag': 'PARAMETERS/sigma',
            'value': sigma.max()
        }])
        if flag and train_config['init_max_step'] < train_config['max_step']:
            train_config['init_max_step'] += 10
        else:
            train_config['init_max_step'] -= 10
        recorder.logger.info('episede: {0} steps: {1} dc_reward: {2} reward: {3}'.format(
            episode, step, total_discounted_reward, total_reward))


def init_or_restore(dicfile, sess, recorder, cp_file):
    if os.path.exists(dicfile + '/checkpoint'):
        try:
            cp_file = tf.train.latest_checkpoint(dicfile)
            begin_episode = int(cp_file.split('-')[-1])
            recorder.saver.restore(sess, cp_file)
        except:
            recorder.logger.error('restore model from checkpoint FAILED.')
        else:
            recorder.logger.info('restore model from checkpoint SUCCUESS.')
            # chkp.recorder.logger.info_tensors_in_checkpoint_file(tf.train.latest_checkpoint(cp_dir), tensor_name='', all_tensors=True)
    else:
        sess.run(tf.global_variables_initializer())
    if 'begin_episode' not in locals().keys():
        begin_episode = 0
    return begin_episode


def writer_summary(writer, x, ys):
    if writer is not None:
        writer.add_summary(tf.Summary(
            value=[
                tf.Summary.Value(tag=y['tag'], simple_value=y['value']) for y in ys
            ]), x)


def export_model(cp_dir, graph):
    """
    Exports latest saved model to .nn format for Unity embedding.
    """
    with graph.as_default():
        target_nodes = ','.join(_process_graph(graph))
        freeze_graph.freeze_graph(
            input_graph=cp_dir + '/raw_graph_def.pb',
            input_binary=True,
            input_checkpoint=tf.train.latest_checkpoint(cp_dir),
            output_node_names=target_nodes,
            # output_graph=(cp_dir + '/frozen_graph_def.pb'),
            output_graph=(cp_dir + '/model.bytes'),
            clear_devices=True, initializer_nodes='', input_saver='',
            restore_op_name='save/restore_all',
            filename_tensor_name='save/Const:0')
    # tf2bc.convert(cp_dir + '/frozen_graph_def.pb', cp_dir + '.nn')


def _process_graph(graph):
    """
    Gets the list of the output nodes present in the graph for inference
    :return: list of node names
    """
    all_nodes = [x.name for x in graph.as_graph_def().node]
    nodes = [x for x in all_nodes if x in possible_output_nodes]
    return nodes
    # return all_nodes


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


def create_logger(name, console_level, console_format, logger2file, file_name, file_level, file_format):
    logger = logging.Logger(name)
    logger.setLevel(level=console_level)
    stdout_handle = logging.StreamHandler(stream=sys.stdout)
    stdout_handle.setFormatter(logging.Formatter(console_format if console_level>20 else '%(message)s'))
    logger.addHandler(stdout_handle)
    if logger2file:
        logfile_handle = logging.FileHandler(file_name)
        logfile_handle.setLevel(file_level)
        logfile_handle.setFormatter(logging.Formatter(file_format))
        logger.addHandler(logfile_handle)
    return logger


if __name__ == '__main__':
    main()
