import sys
from mlagents.envs import UnityEnvironment
from pg2 import PolicyGradient
import numpy as np

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

            trainer.learn()

            break