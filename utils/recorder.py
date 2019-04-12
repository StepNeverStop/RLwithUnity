import sys
import tensorflow as tf
import pandas as pd
from pymongo import MongoClient

class Recorder(object):
    def __init__(self, log_dir, excel_dir, record_config, logger, max_to_keep=5, pad_step_number=True, graph=None):
        self.saver = tf.train.Saver(max_to_keep=max_to_keep, pad_step_number=pad_step_number)
        self.writer = writer=tf.summary.FileWriter(log_dir, graph = graph)
        self.excel_writer = pd.ExcelWriter(excel_dir + '/data.xlsx')
        self.mongocon = MongoClient('127.0.0.1', 27017)
        self.mongodb = self.mongocon[
            record_config['project_name'] + r'_' \
            + record_config['remark'] \
            + record_config['run_id']
        ]
        self.logger = logger
        
    def close(self):
        self.mongocon.close()
