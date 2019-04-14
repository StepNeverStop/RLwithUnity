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


class SAC(object):
    def __init__(self, sess, s_dim, a_counts, hyper_config):
        self.sess = sess
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.activation_fn = tf.nn.tanh
        self.log_alpha = tf.get_variable(
            'log_alpha', dtype=tf.float32, initializer=0.0)
        self.alpha = hyper_config['alpha'] if not hyper_config['auto_adaption'] else tf.exp(
            self.log_alpha)
        self.action_bound = hyper_config['action_bound']
        self.assign_interval = hyper_config['assign_interval']

        self.episode = tf.Variable(tf.constant(0))
        self.lr = tf.train.polynomial_decay(
            hyper_config['lr'], self.episode, hyper_config['max_episode'], 1e-10, power=1.0)
        self.s = tf.placeholder(tf.float32, [None, self.s_dim], 'state')
        self.a = tf.placeholder(tf.float32, [None, self.a_counts], 'action')
        self.r = tf.placeholder(tf.float32, [None, 1], 'reward')
        self.s_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state')
        self.sigma_offset = tf.placeholder(
            tf.float32, [self.a_counts, ], 'sigma_offset')

        self.norm_dist, self.a_new, self.log_prob = self._build_actor_net(
            'actor_net')
        self.prob = self.norm_dist.prob(self.a_new)
        self.entropy = self.norm_dist.entropy()
        self.s_a = tf.concat((self.s, self.a), axis=1)
        self.s_a_new = tf.concat((self.s, self.a_new), axis=1)
        self.q1 = self._build_q_net('q1', self.s_a, False)
        self.q2 = self._build_q_net('q2', self.s_a, False)
        self.q1_anew = self._build_q_net('q1', self.s_a_new, True)
        self.q2_anew = self._build_q_net('q2', self.s_a_new, True)
        self.v_from_q = tf.minimum(
            self.q1_anew, self.q2_anew) - self.alpha * self.log_prob
        self.v_from_q_stop = tf.stop_gradient(self.v_from_q)
        self.v, self.v_var = self._build_v_net(
            'v', input_vector=self.s, trainable=True)
        self.v_target, self.v_target_var = self._build_v_net(
            'v_target', input_vector=self.s_, trainable=False)
        self.dc_r = tf.stop_gradient(
            self.r + hyper_config['gamma'] * self.v_target)

        self.q1_loss = tf.reduce_mean(
            tf.squared_difference(self.q1, self.dc_r))
        self.q2_loss = tf.reduce_mean(
            tf.squared_difference(self.q2, self.dc_r))
        self.v_loss = tf.reduce_mean(
            tf.squared_difference(self.v, self.v_from_q))
        self.v_loss_stop = tf.reduce_mean(
            tf.squared_difference(self.v, self.v_from_q_stop))
        self.critic_loss = 0.5 * self.q1_loss + 0.5 * \
            self.q2_loss + 0.5 * self.v_loss_stop
        self.actor_loss = -tf.reduce_mean(
            self.q1_anew - self.alpha * self.log_prob)
        self.alpha_loss = -tf.reduce_mean(
            self.log_alpha * tf.stop_gradient(self.log_prob - self.a_counts))

        q1_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='q1')
        q2_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='q2')
        value_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='v')
        actor_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor_net')

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_q1 = optimizer.minimize(self.q1_loss, var_list=q1_vars)
        self.train_q2 = optimizer.minimize(self.q2_loss, var_list=q2_vars)
        self.train_v = optimizer.minimize(self.v_loss, var_list=value_vars)

        self.assign_v_target = tf.group([tf.assign(
            r, hyper_config['ployak'] * v + (1 - hyper_config['ployak']) * r) for r, v in zip(self.v_target_var, self.v_var)])
        # self.assign_v_target = [
        #     tf.assign(r, 1/(self.episode+1) * v + (1-1/(self.episode+1)) * r) for r, v in zip(self.v_target_var, self.v_var)]
        with tf.control_dependencies([self.assign_v_target]):
            self.train_critic = optimizer.minimize(self.critic_loss)
        with tf.control_dependencies([self.train_critic]):
            self.train_actor = optimizer.minimize(
                self.actor_loss, var_list=actor_vars)
        with tf.control_dependencies([self.train_actor]):
            self.train_alpha = optimizer.minimize(
                self.alpha_loss, var_list=[self.log_alpha])
        

    def _build_actor_net(self, name):
        with tf.variable_scope(name):
            actor1 = tf.layers.dense(
                inputs=self.s,
                units=128,
                activation=self.activation_fn,
                name='actor1',
                **initKernelAndBias
            )
            actor2 = tf.layers.dense(
                inputs=actor1,
                units=64,
                activation=self.activation_fn,
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
                activation=self.activation_fn,
                name='simga1',
                **initKernelAndBias
            )
            self.sigma = tf.layers.dense(
                inputs=sigma1,
                units=self.a_counts,
                activation=tf.nn.sigmoid,
                name='sigma',
                **initKernelAndBias
            )
            norm_dist = tf.distributions.Normal(
                loc=self.mu, scale=self.sigma + self.sigma_offset)
            # action = tf.tanh(norm_dist.sample())
            action = tf.clip_by_value(
                norm_dist.sample(), -self.action_bound, self.action_bound) / self.action_bound
            log_prob = norm_dist.log_prob(action)
        return norm_dist, action, log_prob

    def _build_q_net(self, name, input_vector, reuse):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=input_vector,
                units=256,
                activation=self.activation_fn,
                name='layer1',
                reuse=reuse,
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=256,
                activation=self.activation_fn,
                name='layer2',
                reuse=reuse,
                **initKernelAndBias
            )
            q = tf.layers.dense(
                inputs=layer2,
                units=1,
                activation=None,
                name='q_value',
                reuse=reuse,
                **initKernelAndBias
            )
        return q

    def _build_v_net(self, name, input_vector, trainable):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=input_vector,
                units=256,
                activation=self.activation_fn,
                name='layer1',
                trainable=trainable,
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=256,
                activation=self.activation_fn,
                name='layer2',
                trainable=trainable,
                **initKernelAndBias
            )
            v = tf.layers.dense(
                inputs=layer2,
                units=1,
                activation=None,
                name='value',
                trainable=trainable,
                **initKernelAndBias
            )
            var = tf.get_variable_scope().global_variables()
        return v, var

    def decay_lr(self, episode, **kargs):
        return self.sess.run(self.lr, feed_dict={
            self.episode: episode
        })

    def choose_action(self, s, sigma_offset, **kargs):
        return self.sess.run([self.prob, self.a_new], feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def choose_inference_action(self, s, sigma_offset, **kargs):
        return self.sess.run([self.prob, self.mu], feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def get_state_value(self, s, sigma_offset, **kargs):
        return np.zeros(np.array(s).shape[0])

    def learn(self, s, a, r, s_, episode, sigma_offset, **kargs):
        # self.sess.run([self.assign_v_target, self.train_q1, self.train_q2, self.train_v, self.train_actor], feed_dict={
        #     self.s: s,
        #     self.a: a,
        #     self.r: r,
        #     self.s_: s_,
        #     self.episode: episode,
        #     self.sigma_offset: sigma_offset
        # })
        self.sess.run([self.assign_v_target, self.train_critic, self.train_actor, self.train_alpha], feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.episode: episode,
            self.sigma_offset: sigma_offset
        })

    def get_entropy(self, s, sigma_offset, **kargs):
        return self.sess.run(self.entropy, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def get_actor_loss(self, s, sigma_offset, **kargs):
        return self.sess.run(self.actor_loss, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })

    def get_critic_loss(self, s, a, r, s_, sigma_offset, **kargs):
        return self.sess.run(self.critic_loss, feed_dict={
            self.s: s,
            self.a: a,
            self.r: r,
            self.s_: s_,
            self.sigma_offset: sigma_offset
        })

    def get_sigma(self, s, sigma_offset, **kargs):
        return self.sess.run(self.sigma, feed_dict={
            self.s: s,
            self.sigma_offset: sigma_offset
        })
