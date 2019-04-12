import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as c_layers

initKernelAndBias = {
    'kernel_initializer': tf.random_normal_initializer(0., .1),
    'bias_initializer': tf.constant_initializer(0.1, dtype=tf.float32)
}
# initKernelAndBias={
#     'kernel_initializer' : c_layers.variance_scaling_initializer(1.0)
# }


class PPO_SEP(object):
    def __init__(self, sess, s_dim, a_counts, hyper_config, action_bound, epsilon, actor_lr, critic_lr, decay_steps, decay_rate=0.95, stair=False):
        self.actor = self.Actor(
            sess=sess,
            s_dim=s_dim,
            a_counts=a_counts,
            decay_rate=hyper_config['decay_rate'],
            decay_steps=hyper_config['decay_steps'],
            action_bound=hyper_config['action_bound'],
            actor_lr=hyper_config['actor_lr'],
            epsilon=hyper_config['epsilon'],
            stair=hyper_config['stair']
        )
        self.critic = self.Critic(
            sess=sess,
            s_dim=s_dim,
            decay_rate=hyper_config['decay_rate'],
            decay_steps=hyper_config['decay_steps'],
            critic_lr=hyper_config['critic_lr'],
            stair=hyper_config['stair']
        )

    def decay_lr(self, episode):
        return self.actor.actor_decay_lr(episode)

    def choose_action(self, state):
        return self.actor.choose_action(state)

    def choose_inference_action(self, state):
        return self.actor.choose_inference_action(state)

    def get_actor_loss(self, s, a, old_prob, advantage):
        return self.actor.get_actor_loss(s, a, old_prob, advantage)

    def get_entropy(self, s):
        return self.actor.get_entropy(s)

    def get_sigma(self, s):
        return self.actor.get_sigma(s)

    def get_state_value(self, state):
        return self.critic.get_state_value(state)

    def get_critic_loss(self, s, dc_r, **kargs):
        return self.critic.get_critic_loss(s, dc_r)

    def learn(self, s, a, dc_r, old_prob, advantage, episode):
        self.actor.learn(s, a, old_prob, advantage, episode)
        self.critic.learn(s, dc_r, episode)

    class Actor(object):
        def __init__(self, sess, s_dim, a_counts, decay_rate, decay_steps, action_bound, actor_lr, epsilon, stair=False):
            self.sess = sess

            self.s_dim = s_dim
            self.a_counts = a_counts
            self.action_bound = tf.constant(action_bound, dtype=tf.float32)
            self.s = tf.placeholder(tf.float32, [None, s_dim], "state")
            self.a = tf.placeholder(tf.float32, [None, a_counts], "action")
            self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
            self.old_prob = tf.placeholder(
                tf.float32, [None, self.a_counts], 'old_prob')
            self.episode = tf.Variable(tf.constant(0))

            self.norm_dist = self._build_actor_net('ActorNet', trainable=True)

            self.sample_op = self.norm_dist.sample()
            with tf.name_scope('out'):
                clip_action = tf.clip_by_value(
                    self.sample_op, -self.action_bound, self.action_bound) / self.action_bound
                self.clip_action = tf.identity(clip_action, name='Action')
            self.prob = self.norm_dist.prob(self.clip_action)
            self.new_prob = self.norm_dist.prob(self.a)
            self.entropy = self.norm_dist.entropy()
            # ratio = self.new_prob / self.old_prob
            ratio = tf.exp(self.new_prob - self.old_prob)
            surrogate = ratio * self.advantage
            self.actor_loss = tf.reduce_mean(tf.minimum(
                surrogate,
                tf.clip_by_value(ratio, 1. - epsilon, 1.
                                 + epsilon) * self.advantage
            ))
            self.decay = tf.train.exponential_decay(
                actor_lr, self.episode, decay_steps, decay_rate, staircase=stair)
            self.train_op = tf.train.AdamOptimizer(
                self.decay).minimize(-self.actor_loss)
            # self.decay = tf.train.inverse_time_decay(self.learning_rate, self.episode, decaystep, decay_rate, staircase=stair)
            # self.decay = tf.train.natural_exp_decay(self.learning_rate, self.episode, decay_steps, decay_rate, staircase=stair)

            # self.gradients = tf.gradients(self.actor_loss,self.new_actor_vars)

        def _build_actor_net(self, name, trainable):
            with tf.variable_scope(name):
                layer1 = tf.layers.dense(
                    inputs=self.s,
                    units=256,
                    activation=tf.nn.relu,
                    name='layer1',
                    **initKernelAndBias,
                    trainable=trainable
                )
                layer2 = tf.layers.dense(
                    inputs=layer1,
                    units=256,
                    activation=tf.nn.relu,
                    name='layer2',
                    **initKernelAndBias,
                    trainable=trainable
                )
                self.mu = tf.layers.dense(
                    inputs=layer2,
                    units=self.a_counts,
                    activation=tf.nn.tanh,
                    name='mu',
                    **initKernelAndBias,
                    trainable=trainable
                )
                self.sigma = tf.layers.dense(
                    inputs=layer2,
                    units=self.a_counts,
                    activation=tf.nn.softplus,
                    name='delta',
                    **initKernelAndBias,
                    trainable=trainable
                )
                norm_dist = tf.distributions.Normal(
                    loc=self.mu, scale=self.sigma + .1)
                # var = tf.get_variable_scope().global_variables()
                return norm_dist

        def choose_action(self, s):
            return self.sess.run([self.clip_action, self.prob], feed_dict={
                self.s: s
            })  # clip action to aviod the action value choosed is too large to cause 'nan' value

        def choose_inference_action(self, s):
            return self.sess.run([self.clip_action, self.prob], feed_dict={
                self.s: s
            })

        def get_entropy(self, s):
            return self.sess.run(self.entropy, feed_dict={
                self.s: s
            })

        def get_actor_loss(self, s, a, old_prob, advantage):
            return self.sess.run(self.actor_loss, feed_dict={
                self.s: s,
                self.a: a,
                self.old_prob: old_prob,
                self.advantage: advantage
            })

        def learn(self, s, a, old_prob, advantage, episode):
            return self.sess.run(self.train_op, feed_dict={
                self.s: s,
                self.a: a,
                self.old_prob: old_prob,
                self.advantage: advantage,
                self.episode: episode
            })

        def get_sigma(self, s):
            return self.sess.run(self.sigma, feed_dict={
                self.s: s
            })

        def actor_decay_lr(self, episode):
            return self.sess.run(self.decay, feed_dict={
                self.episode: episode
            })
            '''
            调试出现nan的问题
            '''
            # print('-----------')
            # print(self.sess.run(self.old_action_prob, feed_dict={
            #     self.s : s,
            #     self.a : a
            # }))
            # aa = self.sess.run(self.old_action_prob, feed_dict={
            #     self.s : s,
            #     self.a : a
            # })
            # bb = self.sess.run(self.new_action_prob, feed_dict={
            #     self.s : s,
            #     self.a : a
            # })
            # print('--------------------')
            # print(self.sess.run(self.gradients,{
            #     self.s : s,
            #     self.a : a,
            #     self.advantage : advantage
            # }))
            # if(np.isnan(bb).any()):
            #     print(bb)
            #     input()

    class Critic(object):
        def __init__(self, sess, s_dim, decay_rate, decay_steps, critic_lr, stair=False):
            self.sess = sess

            self.s = tf.placeholder(tf.float32, [None, s_dim], "state")
            self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.episode = tf.Variable(tf.constant(0))
            self.learning_rate = tf.Variable(critic_lr, dtype=tf.float32)

            self._build_critic_net('Critic')
            self.loss = tf.reduce_mean(
                tf.squared_difference(self.dc_r, self.v))
            self.decay = tf.identity(self.learning_rate)
            self.train_op = tf.train.AdamOptimizer(
                self.decay).minimize(self.loss)
            # self.decay = tf.train.natural_exp_decay(self.learning_rate, self.episode, decay_steps, decay_rate, staircase=stair)

        def _build_critic_net(self, name):
            with tf.variable_scope(name):
                layer1 = tf.layers.dense(
                    inputs=self.s,
                    units=256,
                    activation=tf.nn.relu,
                    name='layer1',
                    **initKernelAndBias
                )
                layer2 = tf.layers.dense(
                    inputs=layer1,
                    units=256,
                    activation=tf.nn.relu,
                    name='layer2',
                    **initKernelAndBias
                )
                self.v = tf.layers.dense(
                    inputs=layer2,
                    units=1,
                    activation=None,
                    name='values',
                    **initKernelAndBias
                )

        def get_state_value(self, s):
            return np.squeeze(self.sess.run(self.v, feed_dict={
                self.s: s
            }))

        def get_critic_loss(self, s, dc_r):
            return self.sess.run(self.loss, feed_dict={
                self.s: s,
                self.dc_r: dc_r
            })

        def critic_decay_lr(self, episode):
            return self.sess.run(self.decay, feed_dict={
                self.episode: episode
            })

        def learn(self, s, dc_r, episode):
            return self.sess.run(self.train_op, feed_dict={
                self.s: s,
                self.dc_r: dc_r,
                self.episode: episode
            })


class PPO_COM(object):
    def __init__(self, sess, s_dim, a_counts, hyper_config):
        self.sess = sess
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.action_bound = hyper_config['action_bound']
        self.episode = tf.Variable(tf.constant(0))
        self.epsilon = tf.train.polynomial_decay(
            hyper_config['epsilon'], self.episode, hyper_config['max_episode'], 0.1, power=1.0)
        self.beta = tf.train.polynomial_decay(
            hyper_config['beta'], self.episode, hyper_config['max_episode'], 1e-5, power=1.0)
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.a_counts], 'action')
        self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")
        self.sigma_offset = tf.placeholder(
            tf.float32, [self.a_counts, ], 'sigma_offset')
        self.norm_dist = self._build_net('ppo')
        self.old_prob = tf.placeholder(
            tf.float32, [None, self.a_counts], 'old_prob')

        self.new_prob = self.norm_dist.prob(self.a)

        self.sample_op = self.norm_dist.sample()
        with tf.name_scope('out'):
            clip_action = tf.clip_by_value(
                self.sample_op, -self.action_bound, self.action_bound) / self.action_bound
            self.clip_action = tf.identity(clip_action, name='Action')
        self.prob = self.norm_dist.prob(self.clip_action)
        self.entropy = self.norm_dist.entropy()
        self.mean_entropy = tf.reduce_mean(self.norm_dist.entropy())
        # ratio = tf.exp(self.new_prob - self.old_prob)
        ratio = self.new_prob / self.old_prob
        surrogate = ratio * self.advantage
        self.actor_loss = tf.reduce_mean(tf.minimum(
            surrogate,
            tf.clip_by_value(ratio, 1.0 - self.epsilon, 1.0
                             + self.epsilon) * self.advantage
        ))
        self.value_loss = tf.reduce_mean(
            tf.squared_difference(self.dc_r, self.value))
        self.loss = -(self.actor_loss - 1.0 * self.value_loss
                      + self.beta * self.mean_entropy)
        self.lr = tf.train.polynomial_decay(
            hyper_config['lr'], self.episode, hyper_config['max_episode'], 1e-10, power=1.0)
        # self.lr = tf.train.exponential_decay(hyper_config['lr'], self.episode, hyper_config['decay_steps'], hyper_config['decay_rate'], staircase=hyper_config['stair'])
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    @staticmethod
    def swish(input_activation):
        """Swish activation function. For more info: https://arxiv.org/abs/1710.05941"""
        return tf.multiply(input_activation, tf.nn.sigmoid(input_activation))

    def _build_net(self, name):
        # activation_fn = self.swish
        activation_fn = tf.nn.tanh
        with tf.variable_scope(name):
            share1 = tf.layers.dense(
                inputs=self.s,
                units=512,
                activation=activation_fn,
                name='share1',
                **initKernelAndBias
            )
            share2 = tf.layers.dense(
                inputs=share1,
                units=256,
                activation=activation_fn,
                name='share2',
                **initKernelAndBias
            )
            actor1 = tf.layers.dense(
                inputs=share2,
                units=128,
                activation=activation_fn,
                name='actor1',
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=activation_fn,
                name='actor2',
                **initKernelAndBias
            )
            self.mu = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                **initKernelAndBias
            )
            sigma1 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=activation_fn,
                name='sigma1',
                **initKernelAndBias
            )
            self.sigma = tf.layers.dense(
                inputs=sigma1,
                units=self.a_counts,
                activation=tf.nn.sigmoid,
                name='sigma',
                **initKernelAndBias
            )
            critic1 = tf.layers.dense(
                inputs=share2,
                units=128,
                activation=activation_fn,
                name='critic1',
                **initKernelAndBias
            )
            critic2 = tf.layers.dense(
                inputs=critic1,
                units=64,
                activation=activation_fn,
                name='critic2',
                **initKernelAndBias
            )
            self.value = tf.layers.dense(
                inputs=critic2,
                units=1,
                activation=None,
                name='value',
                **initKernelAndBias
            )
        norm_dist = tf.distributions.Normal(
            loc=self.mu, scale=self.sigma + self.sigma_offset)
        return norm_dist

    def learn(self, s, a, dc_r, old_prob, advantage, episode, sigma_offset):
        self.sess.run(self.train_op, feed_dict={
            self.s: s,
            self.a: a,
            self.dc_r: dc_r,
            self.old_prob: old_prob,
            self.advantage: advantage,
            self.episode: episode,
            self.sigma_offset: sigma_offset
        })

    def choose_action(self, s, sigma_offset):
        return self.sess.run([self.prob, self.clip_action], feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def choose_inference_action(self, s, sigma_offset):
        return self.sess.run([self.prob, self.clip_action], feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def print_prob(self, s, a, sigma_offset):
        print(self.sess.run(self.new_prob, feed_dict={
            self.s: s,
            self.a: a,
            self.sigma_offset: sigma_offset
        }))

    def decay_lr(self, episode):
        return self.sess.run(self.lr, feed_dict={
            self.episode: episode
        })

    def get_state_value(self, s, sigma_offset):
        return np.squeeze(self.sess.run(self.value, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        }))

    def get_entropy(self, s, sigma_offset):
        return self.sess.run(self.entropy, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def get_actor_loss(self, s, a, old_prob, advantage, sigma_offset):
        return self.sess.run(self.actor_loss, feed_dict={
            self.s: s,
            self.a: a,
            self.old_prob: old_prob,
            self.advantage: advantage,
            self.sigma_offset: sigma_offset
        })

    def get_critic_loss(self, s, dc_r, sigma_offset, **kargs):
        return self.sess.run(self.value_loss, feed_dict={
            self.s: s,
            self.dc_r: dc_r,
            self.sigma_offset: sigma_offset
        })

    def get_sigma(self, s, sigma_offset):
        return self.sess.run(self.sigma, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })
