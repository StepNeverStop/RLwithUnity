import numpy as np
import tensorflow as tf
from mlagents.envs import UnityEnvironment

np.random.seed(1)
tf.set_random_seed(1)

class PolicyGradient:
    def __init__(
        self,
        actionDim,
        observationDim,
        learning_rate=0.01,
        reward_decay=0.95,
    ):
        self.actionDim=actionDim
        self.observationDim=observationDim
        self.alpha=learning_rate
        self.gamma=reward_decay

        self.observations, self.actions, self.rewards=[],[],[]
        
        self._build_net()

        self.sess=tf.Session()

        self.sess.run(tf.global_variables_initializer())
        
        self.writer=tf.summary.FileWriter("logs/test/")
        

    def _build_net(self):
        with tf.name_scope('Inputs'):
            self.tf_obs = tf.placeholder(tf.float32, [None, self.observationDim], name="observations")
            self.tf_acts = tf.placeholder(tf.float32, [None, self.actionDim], name="actions")
            self.tf_vt = tf.placeholder(tf.float32, [None, 1],name="actions_value")

        layer = tf.layers.dense(
            inputs=self.tf_obs,
            units=10,
            activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1'
        )

        self.all_act=tf.layers.dense(
            inputs=layer,
            units=self.actionDim,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0,stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer2'
        )

        self.norm_dist = tf.distributions.Normal(loc=self.all_act,scale=[1., 1.])

        # self.sample_op = tf.squeeze(self.norm_dist.sample(1),axis=0)
        self.sample_op = self.norm_dist.sample()

        log_act_prob=self.norm_dist.log_prob(self.tf_acts)
        loss=tf.reduce_mean(log_act_prob*self.tf_vt)
        tf.summary.scalar('loss',loss)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.alpha).minimize(-loss)
        
        self.merged = tf.summary.merge_all()

    def choose_action(self,obs):
        return self.sess.run(self.sample_op,feed_dict={
            self.tf_obs:obs[np.newaxis,:]
        })

    def store_transition(self, obs, action, reward):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)

    def learn(self,episode):
        discount_returns=self._discount_returns()
        # print(discount_returns)
        summary,_=self.sess.run([self.merged,self.train_op], feed_dict={
            self.tf_obs: np.array(self.observations),
            self.tf_acts: np.array(self.actions),
            self.tf_vt: discount_returns,
        })
        self.writer.add_summary(summary,episode)
        self.observations, self.actions, self.rewards=[],[],[]

    def _discount_returns(self):
        discount_returns = np.zeros_like(self.rewards)
        running_add=0
        for i in reversed(range(0,len(self.rewards))):
            running_add=running_add*self.gamma + self.rewards[i]
            discount_returns[i]=running_add
        return discount_returns[:,np.newaxis]


env = UnityEnvironment()
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
print(brain.vector_observation_space_size)
print(brain.vector_action_space_size)

trainer = PolicyGradient(
    actionDim=brain.vector_action_space_size[0],
    observationDim=brain.vector_observation_space_size,
    learning_rate=0.02,
    reward_decay=0.99,
)

# Value=[1,-1,-1,1]
for i_episode in range(5000):

    observation = env.reset(train_mode=True)[brain_name]
    while True:

        action = trainer.choose_action(observation.vector_observations[0])
        # print(action)
        # 离散
        # actionValue=np.zeros((trainer.n_actions,), dtype=int)
        # actionValue[action]=Value[action]
        # #print(actionValue)
        # observation=env.step(actionValue)[brain_name]

        # 连续
        observation=env.step(action)[brain_name]
        trainer.store_transition(observation.vector_observations[0],np.squeeze(action,axis=0), observation.rewards[0])

        if observation.local_done[0]:
            ep_rs_sum = sum(trainer.rewards)
            print("episode:", i_episode, "  reward:", int(ep_rs_sum))

            trainer.learn(i_episode)
            break