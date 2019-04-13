import numpy as np

class ReplayBuffer(object):

    def __init__(self, s_dim, a_counts, buffer_size):
        self.state = np.zeros([buffer_size, s_dim], dtype=np.float32)
        self.next_state = np.zeros([buffer_size, s_dim], dtype=np.float32)
        self.action = np.zeros([buffer_size, a_counts], dtype=np.float32)
        self.prob = np.zeros([buffer_size, a_counts], dtype=np.float32)
        self.reward = np.zeros(buffer_size, dtype=np.float32)
        self.discounted_reward = np.zeros(buffer_size, dtype=np.float32)
        self.td_error = np.zeros(buffer_size, dtype=np.float32)
        self.advantage = np.zeros(buffer_size, dtype=np.float32)
        self.done = np.zeros(buffer_size, dtype=np.float32)
        self.now, self.buffer_size, self.max_size = 0, 0, buffer_size

    def store(self, state, action, prob, reward, discounted_reward, td_error, advantage, next_state, done):
        self.state[self.now] = state
        self.next_state[self.now] = next_state
        self.action[self.now] = action
        self.prob[self.now] = prob
        self.reward[self.now] = reward
        self.discounted_reward[self.now] = discounted_reward
        self.td_error[self.now] = td_error
        self.advantage[self.now] = advantage
        self.done[self.now] = done
        self.now = (self.now+1) % self.max_size
        self.buffer_size = min(self.buffer_size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        indexs = np.random.randint(0, self.buffer_size, size=batch_size)
        return dict(state=self.state[indexs],
                    next_state=self.next_state[indexs],
                    action=self.action[indexs],
                    old_prob=self.prob[indexs],
                    reward=self.reward[indexs],
                    discounted_reward=self.discounted_reward[indexs],
                    td_error=self.td_error[indexs],
                    advantage=self.advantage[indexs],
                    done=self.done[indexs])