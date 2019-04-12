import os
import sys
import shutil
from config_file import config


def clean(config):
    length = len(config['clean_list'])
    count = 0
    for i in range(length):
        cp_dir = config['record config']['checkpoint_basic_dir'] + \
            config['clean_list'][i]
        log_dir = config['record config']['log_basic_dir'] + \
            config['clean_list'][i]
        excel_dir = config['record config']['excel_basic_dir'] + \
            config['clean_list'][i]
        config_dir = config['record config']['config_basic_dir'] + \
            config['clean_list'][i]
        if os.path.exists(log_dir):
            print('-' * 10 + str(i) + '-' * 10)
            print('C.L.E.A.N : {0}'.format(config['clean_list'][i]))
            try:
                shutil.rmtree(log_dir)
                print('remove LOG success.')
                shutil.rmtree(cp_dir)
                print('remove CHECKPOINT success.')
                shutil.rmtree(excel_dir)
                print('remove EXCEL success.')
                shutil.rmtree(config_dir)
                print('remove CONFIG success.')
                count += 1
            except Exception as e:
                print(e)
                sys.exit()
        else:
            print('{0} is not exist, please check and run again...'.format(
                config['clean_list'][i]))
    print(f'total: {length}, clean: {count}')


clean(config)
