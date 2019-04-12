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

class DDPG(object):
    def __init__(self, sess, s_dim, a_counts, hyper_config):
        self.sess = sess
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.activation_fn = tf.nn.relu
        self.action_bound = hyper_config['action_bound']
        self.assign_interval = hyper_config['assign_interval']

        self.episode = tf.Variable(tf.constant(0))
        self.lr = tf.train.polynomial_decay(
            hyper_config['lr'], self.episode, hyper_config['max_episode'], 1e-10, power=1.0)
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.a_counts], 'action')
        self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_reward')

        self.mu, self.action, self.actor_var = self._build_actor_net(
            'actor', True)
        self.target_mu, self.action_target, self.actor_target_var = self._build_actor_net(
            'actor_target', False)

        self.s_a = tf.concat((self.s, self.a), axis=1)
        self.s_mu = tf.concat((self.s, self.mu), axis=1)
        self.s_a_target = tf.concat((self.s, self.target_mu), axis=1)
        
        self.q, self.q_var = self._build_q_net(
            'q', self.s_a, True, reuse=False)
        self.q_actor, _ = self._build_q_net('q', self.s_mu, True, reuse=True)
        self.q_target, self.q_target_var = self._build_q_net(
            'q_target', self.s_a_target, False, reuse=False)

        self.q_loss = tf.reduce_mean(
            tf.squared_difference(self.q, self.dc_r))
        self.actor_loss = -tf.reduce_mean(self.q_actor)

        q_var = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='q')
        actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_q = optimizer.minimize(self.q_loss, var_list=q_var)
        with tf.control_dependencies([self.train_q]):
            self.train_actor = optimizer.minimize(
                self.actor_loss, var_list=actor_vars)
        with tf.control_dependencies([self.train_actor]):
            self.assign_q_target = tf.group([tf.assign(
                r, hyper_config['ployak'] * v + (1 - hyper_config['ployak']) * r) for r, v in zip(self.q_target_var, self.q_var)])
            self.assign_actor_target = tf.group([tf.assign(
                r, hyper_config['ployak'] * v + (1 - hyper_config['ployak']) * r) for r, v in zip(self.actor_target_var, self.actor_var)])
        # self.assign_q_target = [
        #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.q_target_var, self.q_var)]
        # self.assign_q_target = [
        #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.actor_target_var, self.actor_var)]

    def _build_actor_net(self, name, trainable):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=self.s,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                trainable=trainable,
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
                name='actor2',
                trainable=trainable,
                **initKernelAndBias
            )
            mu = tf.layers.dense(
                inputs=actor2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                trainable=trainable,
                **initKernelAndBias
            )
            e = tf.random_normal(tf.shape(mu))
            action = tf.clip_by_value(
                mu + e, -self.action_bound, self.action_bound) / self.action_bound
            var = tf.get_variable_scope().global_variables()
        return mu, action, var

    def _build_q_net(self, name, input_vector, trainable, reuse=False):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=input_vector,
                units=256,
                activation=self.activation_fn,
                name='layer1',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=256,
                activation=self.activation_fn,
                name='layer2',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            q = tf.layers.dense(
                inputs=layer2,
                units=1,
                activation=None,
                name='q_value',
                trainable=trainable,
                reuse=reuse,
                **initKernelAndBias
            )
            var = tf.get_variable_scope().global_variables()
        return q, var

    def decay_lr(self, episode, **kargs):
        return self.sess.run(self.lr, feed_dict={
            self.episode: episode
        })

    def choose_action(self, s, **kargs):
        return np.ones(s.shape), self.sess.run(self.action, feed_dict={
            self.s: s
        })

    def choose_inference_action(self, s, **kargs):
        return np.ones(s.shape), self.sess.run(self.mu, feed_dict={
            self.s: s
        })

    def get_state_value(self, s, **kargs):
        return np.squeeze(
            self.sess.run(self.q_target, feed_dict={
                self.s: s
            }))

    def learn(self, s, a, dc_r, episode, **kargs):
        self.sess.run([self.train_q, self.train_actor, self.assign_q_target, self.assign_actor_target], feed_dict={
            self.s: s,
            self.a: a,
            self.dc_r: dc_r,
            self.episode: episode
        })

    def get_actor_loss(self, s, **kargs):
        return self.sess.run(self.actor_loss, feed_dict={
            self.s: s
        })

    def get_critic_loss(self, s, a, dc_r, **kargs):
        return self.sess.run(self.q_loss, feed_dict={
            self.s: s,
            self.a: a,
            self.dc_r: dc_r
            })

    def get_entropy(self, s, **kargs):
        return np.zeros(np.array(s).shape)

    def get_sigma(self, s, **kargs):
        return np.zeros(np.array(s).shape)
