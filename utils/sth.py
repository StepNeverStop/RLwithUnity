import os
import yaml
import numpy as np
import tensorflow as tf

class sth(object):
    @staticmethod
    def discounted_sum(x, gamma, init_value, done_index, length=0, z=None):
        """
        x: list of data
        gamma: discounted factor
        init_value: the initiate value of the last data
        return: a list of discounted numbers
        
        Examples:
        x: [1, 2, 3]
        gamma: 0.5
        init_value: 0.0
        return: [2.75, 3.5, 3]
        """
        y = []
        if length == 0:
            length = len(x)
        if len(done_index) > 0:
            for i in reversed(range(length)):
                for j in reversed(done_index):
                    if i == j:
                        init_value = 0
                init_value = init_value * gamma + x[i]
                y.append(init_value)
                if z is not None:
                    init_value = z[i]
        else:
            for i in reversed(range(length)):
                init_value = init_value * gamma + x[i]
                y.append(init_value)
                if z is not None:
                    init_value = z[i]
        y.reverse()
        return y

    @staticmethod
    def discounted_sum_minus(x, gamma, init_value, done_index, z, length=0):
        """
        x: list of data
        gamma: discounted factor
        init_value: the initiate value of the last data
        return: a list of discounted numbers
        :
        Examples:
        x: [1, 2, 3]
        gamma: 0.5
        init_value: 0.0
        z: [1, 2, 3]
        return: [1, 1.5, 0]
        """
        y = []
        if length == 0:
            length = len(x)
        if len(done_index) > 0:
            for i in reversed(range(length)):
                for j in reversed(done_index):
                    if i == j:
                        init_value =0
                y.append(init_value * gamma + x[i] - z[i])
                init_value = z[i]
        else:
            for i in reversed(range(length)):
                y.append(init_value * gamma + x[i] - z[i])
                init_value = z[i]
        y.reverse()
        return y

    @staticmethod
    def get_discounted_sum(x, gamma, init_value, length=0):
        """
        x: list of data
        gamma: discounted factor
        init_value: the initiate value of the last data
        return: the value of discounted sum, type 1-D
        :
        Examples:
        x: [1, 2, 3]
        gamma: 1
        init_value: 0.0
        return: 6
        """
        if length == 0:
            length = len(x)
        for i in reversed(range(length)):
            init_value = init_value * gamma + x[i]
        return init_value

    @staticmethod
    def split_batchs(x, batchsize, length=0, cross=False, reverse=True):
        """
        x: list of date
        batchsize: size of each block that be splited
        length: the last index of data
        cross: if TRUE, the index will minus 1, otherwise it will minus batchsize
        :
        Examples:
        x: [1, 2, 3, 4]
        batchsize: 2
        reverse: False
        cross: F    return: [[1,2],[3,4]]
        cross: T    return: [[1,2],[2,3],[3,4]]
        """
        if length == 0:
            length = len(x)
        if length < batchsize:
            return [x]
        if reverse:
            if cross:
                return list(reversed([x[i:i+batchsize] for i in range(length-batchsize+1)]))
            else:
                return list(reversed([x[i:i+batchsize] for i in range(0, length, batchsize)]))
        else:
            if cross:
                return list([x[i:i+batchsize] for i in range(length-batchsize+1)])
            else:
                return list([x[i:i+batchsize] for i in range(0, length, batchsize)])

    @staticmethod
    def check_or_create(dicpath, name=''):
        if not os.path.exists(dicpath):
            os.makedirs(dicpath)
            print(f'create {name} directionary :', dicpath)
    @staticmethod
    def save_config(filename, config):
        fw = open(os.path.join(filename, 'config.yaml'), 'w', encoding='utf-8')
        yaml.dump(config, fw)
        fw.close()
        print(f'save config to {filename}')
    @staticmethod
    def load_config(filename):
        f = open(os.path.join(filename, 'config.yaml'), 'r', encoding='utf-8')
        x = yaml.safe_load(f.read())
        f.close()
        print(f'load config from {filename}')
        return x