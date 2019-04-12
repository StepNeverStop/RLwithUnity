import numpy as np
import tensorflow as tf
from mlagents.envs import UnityEnvironment

initKernelAndBias={
    'kernel_initializer' : tf.random_normal_initializer(0., .1),
    'bias_initializer' : tf.constant_initializer(0.1)
}

def writer_summary(writer, x, ys):
    if writer is not None:
        writer.add_summary(tf.Summary(
            value = [
                tf.Summary.Value(tag = y['tag'], simple_value = y['value']) for y in ys
                ]), x)

class Actor(object):
    def __init__(self, sess, observationDim, actionDim, learning_rate=0.001):
        self.sess=sess
        self.s=tf.placeholder(tf.float32, [1,observationDim],"state")
        self.a=tf.placeholder(tf.float32, [1,actionDim],"action")
        self.advantage=tf.placeholder(tf.float32,[1,1],"advantage")

        with tf.variable_scope("Actor"):
            layer1 = tf.layers.dense(
                inputs=self.s,
                units=20,
                activation=tf.nn.relu,
                name='layer1',
                **initKernelAndBias
            )

            layer2=tf.layers.dense(
                inputs=layer1,
                units=20,
                activation=tf.nn.relu,
                name='layer2',
                **initKernelAndBias
            )

            self.mu = tf.layers.dense(
                inputs=layer2,
                units=actionDim,
                activation=None,
                name='mu',
                **initKernelAndBias
            )

            self.norm_dist = tf.distributions.Normal(loc=self.mu,scale=[1.]*actionDim)
            self.sample_op = self.norm_dist.sample()
        with tf.variable_scope('exp_v'):
            self.log_prob = self.norm_dist.log_prob(self.a)
            self.exp_v = tf.reduce_mean(self.log_prob*self.advantage)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(-self.exp_v)

    def choose_action(self, s):
        return self.sess.run(self.sample_op,feed_dict={
            self.s:s
        })
    def learn(self, s, a, advantage):
        self.sess.run(self.train_op,feed_dict={
            self.s:s,
            self.a:a,
            self.advantage:advantage
        })

class Critic(object):
    def __init__(self, sess, observationDim, learning_rate=0.01, gamma=0.95):
        self.sess= sess

        self.s = tf.placeholder(tf.float32, [1,observationDim],"state")
        self.r = tf.placeholder(tf.float32, [1,1],"reward")
        self.v_ = tf.placeholder(tf.float32, [1,1], "value_of_next")

        with tf.variable_scope('Critic'):
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
                name='Value',
                **initKernelAndBias
            )
        with tf.variable_scope('square_advantage'):
            self.advantage = tf.reduce_mean(self.r + gamma*self.v_-self.v)
            self.loss = tf.square(self.advantage)
        # tf.summary.scalar('advantage',self.advantage)
        tf.summary.scalar('loss_square',self.loss)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def learn(self, s, r, s_, merged):
        v_ = self.sess.run(self.v, feed_dict={
            self.s: s_
        })
        summary, advantage, _ = self.sess.run([merged, self.advantage, self.train_op], feed_dict={
            self.s: s,
            self.v_: v_,
            self.r: r
        })
        return summary, advantage

env = UnityEnvironment()
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain.vector_observation_space_size)
print(brain.vector_action_space_size)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

actor = Actor(
    sess=sess,
    observationDim=brain.vector_observation_space_size,
    actionDim=brain.vector_action_space_size[0],
    learning_rate=0.0001
)
critic = Critic(
    sess=sess,
    observationDim=brain.vector_observation_space_size,
    learning_rate=0.0002,
    gamma=0.95
)

merged = tf.summary.merge_all()
sess.run(tf.global_variables_initializer())

writer=tf.summary.FileWriter("logs/RollerBall/pgac", sess.graph)

time=0
gamma=0.9
for i_episode in range(500000):
    step=0
    discounted_reward=0
    observation = env.reset(train_mode=True)[brain_name]
    s=observation.vector_observations
    while True:
        time+=1
        
        step+=1
        action = np.squeeze(actor.choose_action(s), axis=0) 
        # print(action)

        observation=env.step(action)[brain_name]
        
        reward=np.array(observation.rewards)
        discounted_reward*=gamma
        discounted_reward+=reward[0]
        summary, advantage = critic.learn(s,reward[np.newaxis,:],observation.vector_observations,merged)
        writer.add_summary(summary,time)
        advantage=[[advantage]]
        # print(advantage)
        actor.learn(s,action[np.newaxis,:],advantage)

        s=observation.vector_observations

        if observation.local_done[0]:
            writer_summary(writer, i_episode, [{
                                'tag' : 'Reward/discounted_reward',
                                'value' : discounted_reward
                            }])
            print("episode:", i_episode," steps:", step," rewards:", discounted_reward)
            break

