import numpy as np


class ReplayBuffer(object):
    def __init__(self, s_dim, a_counts, buffer_size, use_priority=False):
        self.use_priority = use_priority
        self.state = np.zeros([buffer_size, s_dim], dtype=np.float32)
        self.next_state = np.zeros([buffer_size, s_dim], dtype=np.float32)
        self.action = np.zeros([buffer_size, a_counts], dtype=np.float32)
        self.prob = np.zeros([buffer_size, a_counts], dtype=np.float32)
        self.reward = np.zeros(buffer_size, dtype=np.float32)
        self.discounted_reward = np.zeros(buffer_size, dtype=np.float32)
        self.sum_tree_n = np.int32(np.ceil(np.log2(buffer_size)))+1
        if self.use_priority:
            self.td_error = [np.zeros(np.int32(np.ceil(buffer_size/np.power(2, i)))) for i in range(self.sum_tree_n)]
        else:
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
        if self.use_priority:
            diff = np.abs(td_error) - self.td_error[0][self.now]
            for i in range(self.sum_tree_n):
                self.td_error[i][self.now // np.power(2, i)] += diff
        else:
            self.td_error[self.now] = td_error
        self.advantage[self.now] = advantage
        self.done[self.now] = done
        self.now = (self.now + 1) % self.max_size
        self.buffer_size = min(self.buffer_size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        if self.use_priority:
            temp_indexs = np.random.random_sample(batch_size) * self.td_error[-1][0]
            indexs = np.zeros(batch_size, dtype=np.int32)
            for index, i in enumerate(temp_indexs):
                k = 0
                for j in reversed(range(self.sum_tree_n-1)):
                    k*=2
                    if self.td_error[j][k] < i:
                        i-=self.td_error[j][k]
                        k+=1
                indexs[index] = k
        else:
            indexs = np.random.randint(0, self.buffer_size, size=batch_size)
        return dict(state=self.state[indexs],
                    next_state=self.next_state[indexs],
                    action=self.action[indexs],
                    old_prob=self.prob[indexs],
                    reward=self.reward[indexs],
                    discounted_reward=self.discounted_reward[indexs],
                    td_error=self.td_error[0][indexs] if self.use_priority else self.td_error[indexs],
                    advantage=self.advantage[indexs],
                    done=self.done[indexs])
    
    def update(self, indexs, td_error):
        for i, j in zip(indexs, td_error):
            if self.use_priority:
                diff = np.abs(j) - self.td_error[0][i]
                for k in range(self.sum_tree_n):
                    self.td_error[k][i // np.power(2, k)] += diff
            else:
                self.td_error[i] = td_error