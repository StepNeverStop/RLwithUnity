import numpy as np
import tensorflow as tf
from mlagents.envs import UnityEnvironment

initKernelAndBias={
    'kernel_initializer' : tf.random_normal_initializer(0., .1),
    'bias_initializer' : tf.constant_initializer(0.1, dtype=tf.float32)
}

EPSILON = 0.2
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001
GAMMA = 0.99
LAMBDA = 1.0
MAX_EP = 5000
MAX_STEP = 200
BATCHSIZE = 4
LEARN_COUNTS = 10

class Actor(object):
    def __init__(self, sess, s_dim, a_counts):
        self.sess = sess
        
        self.s_dim = s_dim
        self.a_counts = a_counts
        self.s = tf.placeholder(tf.float32, [None, s_dim], "state")
        self.a = tf.placeholder(tf.float32, [None, a_counts], "action")
        self.advantage = tf.placeholder(tf.float32, [None, 1], "advantage")

        self.new_norm_dist, self.new_actor_vars = self._build_actor_net('ActorNew', trainable=True)
        self.old_norm_dist, self.old_actor_vars = self._build_actor_net('ActorOld', trainable=False)

        self.sample_op = self.new_norm_dist.sample()
        # self.sample_op = self.old_norm_dist.sample()

        self.old_action_prob = self.old_norm_dist.prob(self.a)
        self.new_action_prob = self.new_norm_dist.prob(self.a)

        ratio = self.new_action_prob / self.old_action_prob
        surrogate = ratio * self.advantage
        self.actor_loss = -tf.reduce_mean(tf.minimum(
            surrogate,
            tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON) * self.advantage
        ))

        self.train_op = tf.train.AdamOptimizer(ACTOR_LEARNING_RATE).minimize(self.actor_loss)

        self.assign_new_to_old =  [p.assign(oldp) for p, oldp in zip(self.new_actor_vars, self.old_actor_vars)]
        # self.assign_new_to_old =  [tf.assign(oldp,p) for p, oldp in zip(self.new_actor_vars, self.old_actor_vars)]

    def _build_actor_net(self, name, trainable):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation=tf.nn.relu,
                name='layer1',
                **initKernelAndBias,
                trainable=trainable
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=20,
                activation=tf.nn.relu,
                name='layer2',
                **initKernelAndBias,
                trainable=trainable
            )
            mu = tf.layers.dense(
                inputs=layer2,
                units=self.a_counts,
                activation=tf.nn.tanh,
                name='mu',
                **initKernelAndBias,
                trainable=trainable
            )
            sigma = tf.layers.dense(
                inputs=layer2,
                units=self.a_counts,
                activation=tf.nn.softplus,
                name='delta',
                **initKernelAndBias,
                trainable=trainable
            )
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
            var = tf.get_variable_scope().global_variables()
            return norm_dist, var

    def choose_action(self, s):
        return self.sess.run(self.sample_op, feed_dict={
            self.s : s
        })

    def learn(self, s, a, advantage):
        advantage = advantage[:, np.newaxis]
        self.sess.run(self.train_op, feed_dict={
            self.s : s,
            self.a : a,
            self.advantage : advantage
        })

    def assign_params(self):
        self.sess.run(self.assign_new_to_old)

class Critic(object):
    def __init__(self, sess, s_dim):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [None, s_dim], "state")
        self.dc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')

        self._build_critic_net('Critic')
        self.advantage = self.dc_r - self.v
        self.loss = tf.reduce_mean(tf.square(self.advantage))
        self.train_op = tf.train.AdamOptimizer(CRITIC_LEARNING_RATE).minimize(self.loss)

    def _build_critic_net(self, name):
        with tf.variable_scope(name):
            layer1 = tf.layers.dense(
                inputs=self.s,
                units=30,
                activation=tf.nn.relu,
                name='layer1',
                **initKernelAndBias
            )
            layer2 = tf.layers.dense(
                inputs=layer1,
                units=10,
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
        return  self.sess.run(self.v, feed_dict={
                self.s : s
                })

    #---------------deprecated
    def get_advantage(self, s):
        self.values = get_state_value(s)
        sub_advantage = tf.zeros_like(self.r)
        advantage = tf.zeros_like(self.r)
        for index in reversed(range(tf.shape(sub_advantage)[0])):
            sub_advantage[index]=self.r[index] + GAMMA * self.values[index+1] - self.values[index]
        tmp = 0
        for index in reversed(range(tf.shape(sub_advantage)[0])):
            tmp = tmp * LAMBDA * GAMMA + sub_advantage[index]
            advantage[index] = tmp
        return advantage
        
    def learn(self, s, dc_r):
        self.sess.run(self.train_op,feed_dict={
            self.s : s,
            self.dc_r : dc_r
        })

def main():
    env = UnityEnvironment()
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    print(brain.vector_observation_space_size)
    print(brain.vector_action_space_size)

    sess = tf.Session()

    actor = Actor(
        sess=sess,
        s_dim=brain.vector_observation_space_size,
        a_counts=brain.vector_action_space_size[0]
    )
    critic = Critic(
        sess=sess,
        s_dim=brain.vector_observation_space_size
    )

    sess.run(tf.global_variables_initializer())
    
    for episode in range(MAX_EP):
        step = 0
        total_reward = 0.
        discounted_reward = 0
        s, a, r, dc_r= [], [], [], []
        obs = env.reset(train_mode=True)[brain_name]
        state = obs.vector_observations
        s.append(state[0])
        while True:
            action = actor.choose_action(state)
            a.append(action[0])
            obs = env.step(action)[brain_name]
            step += 1
            reward = obs.rewards
            r.append(reward[0])
            state = obs.vector_observations
            done = obs.local_done[0]
            if done or step >= MAX_STEP:
                if len(s) < BATCHSIZE:
                    break
                else:
                    length = len(s)
                    for index in reversed(range(length)):
                        discounted_reward = discounted_reward * GAMMA + r[index]
                        dc_r.append(discounted_reward)
                    total_reward = dc_r[-1]
                    s_ = list(reversed([s[index:index+BATCHSIZE] for index in range(length-BATCHSIZE+1)]))
                    a_ = list(reversed([a[index:index+BATCHSIZE] for index in range(length-BATCHSIZE+1)]))
                    r_ = list(reversed([r[index:index+BATCHSIZE] for index in range(length-BATCHSIZE+1)]))
                    dc_r_ = list(reversed([dc_r[index:index+BATCHSIZE] for index in range(length-BATCHSIZE+1)]))
                    for index in range(len(s_)):
                        actor.assign_params()
                        ss = np.array(s_[index])
                        aa = np.array(a_[index])
                        rr = np.array(r_[index])
                        dc_rr =np.array(dc_r_[index])[:, np.newaxis]
                        values = critic.get_state_value(ss)
                        value_ = critic.get_state_value(state)
                        sub_advantage=np.zeros_like(rr)
                        for index in reversed(range(np.shape(rr)[0])):
                            sub_advantage[index] = rr[index] + GAMMA * value_ - values[index]
                            value_ = values[index]
                        tmp = 0
                        advantage=np.zeros_like(sub_advantage)
                        for index in reversed(range(np.shape(sub_advantage)[0])):
                            tmp = tmp * LAMBDA * GAMMA + sub_advantage[index]
                            advantage[index] = tmp

                        [actor.learn(ss, aa, advantage) for _ in range(LEARN_COUNTS)]
                        [critic.learn(ss, dc_rr) for _ in range(LEARN_COUNTS)]
                    if done:
                        break

                s, a, r, dc_r= [], [], [], []
                s.append(state[0])
            else:
                s.append(state[0])
        print('episede: {0} steps: {1} reward: {2}'.format(episode, step, total_reward))
if __name__ == '__main__':
    main()