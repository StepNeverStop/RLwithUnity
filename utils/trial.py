class TP(object):
    def __init__(self, sess, s_dim, a_counts):
        self.sess = sess
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.s_a = tf.placeholder(
            tf.float32, [None, s_dim + a_counts], "stateAndAction")
        self.s_ = tf.placeholder(tf.float32, [None, s_dim], "realTransition")
        self._build_tp_net('Transfer')
        #
        self.loss = tf.reduce_mean(tf.square(self.s_ - self.s))
        self.train_op = tf.train.AdamOptimizer(
            hyper_config['tp_lr']).minimize(self.loss)

    def _build_tp_net(self, name):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=self.s_a,
                units=64,
                activation=tf.nn.relu,
                name='layer1',
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=64,
                activation=tf.nn.relu,
                name='layer2',
                **initKernelAndBias
            )
            layer3 = tf.layers.dense(
                inputs=layer2,
                units=32,
                activation=tf.nn.relu,
                name='layer3',
                **initKernelAndBias
            )
            layer4 = tf.layers.dense(
                inputs=layer3,
                units=16,
                activation=tf.nn.relu,
                name='layer4',
                **initKernelAndBias
            )
            self.s = tf.layers.dense(
                inputs=layer4,
                units=self.s_dim,
                activation=None,
                name='state',
                **initKernelAndBias
            )

    def learn(self, s_a, s_):
        return self.sess.run(self.train_op, feed_dict={
            self.s_a: s_a,
            self.s_: s_
        })

    def get_tp_loss(self, s_a, s_):
        return self.sess.run(tf.reduce_mean(self.loss), feed_dict={
            self.s_a: s_a,
            self.s_: s_
        })


class Reward(object):
    def __init__(self, sess, s_dim, a_counts):
        self.sess = sess
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.s_a = tf.placeholder(
            tf.float32, [None, s_dim + a_counts], "stateAndAction")
        self.r_ = tf.placeholder(tf.float32, [None, 1], "realReward")
        self._build_reward_net('Reward')
        self.loss = tf.reduce_mean(tf.square(self.r_ - self.r))
        self.train_op = tf.train.AdamOptimizer(
            hyper_config['reward_lr']).minimize(self.loss)

    def _build_reward_net(self, name):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=self.s_a,
                units=64,
                activation=tf.nn.relu,
                name='layer1',
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=64,
                activation=tf.nn.relu,
                name='layer2',
                **initKernelAndBias
            )
            layer3 = tf.layers.dense(
                inputs=layer2,
                units=32,
                activation=tf.nn.relu,
                name='layer3',
                **initKernelAndBias
            )
            layer4 = tf.layers.dense(
                inputs=layer3,
                units=16,
                activation=tf.nn.relu,
                name='layer4',
                **initKernelAndBias
            )
            self.r = tf.layers.dense(
                inputs=layer4,
                units=1,
                activation=None,
                name='reward',
                **initKernelAndBias
            )

    def learn(self, s_a, r_):
        return self.sess.run(self.train_op, feed_dict={
            self.s_a: s_a,
            self.r_: r_
        })

    def get_reward_loss(self, s_a, r_):
        return self.sess.run(tf.reduce_mean(self.loss), feed_dict={
            self.s_a: s_a,
            self.r_: r_
        })