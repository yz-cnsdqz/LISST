import yaml
import os, sys
import socket

def get_host_name():
    return hostname

def get_body_model_path():
    if 'vlg-atlas' in hostname:
        bmpath = '/local/home/yanzhang25/body/VPoser'
    elif 'emerald' in hostname:
        bmpath = '/home/yzhang/body/VPoser'
    else:
        bmpath = r"C:\Users\ethuser\Desktop\FAA\MOJO-plus\models\VPoser"
        #raise ValueError('not stored here')
    return bmpath



def get_cmu_canonicalized_path(use_tmp_set=False):
    if 'vlg-atlas' in hostname:
        raise NotImplementedError('not supported yet on vlg-atlas')
    if 'emerald' in hostname:
        if not use_tmp_set:
            mkpath = '/mnt/hdd/datasets/CMU-canon'
        else:
            mkpath = '/mnt/hdd/datasets/CMU-canon-tmp'
    else:
        raise ValueError('not stored here')
    return mkpath


def get_cmu_canonicalizedx8_path(split='all'):
    if 'vlg-atlas' in hostname:
        if split in ['train', 'test']:
            mkpath = '/local/home/yanzhang25/datasets/CMU-canon-MPx8-'+split
        elif split == 'all':
            mkpath = '/local/home/yanzhang25/datasets/CMU-canon-MPx8'
        else:
            raise ValueError("split not supported.")
    elif 'emerald' in hostname:
        if split in ['train', 'test']:
            mkpath = '/mnt/hdd/datasets/CMU-canon-MPx8-'+split
        elif split == 'all':
            mkpath = '/mnt/hdd/datasets/CMU-canon-MPx8'
        else:
            raise ValueError("split not supported.")
        
    else:
        raise ValueError('not stored here')
    return mkpath



def get_tmpfolder_path():
    if 'vlg-atlas' in hostname:
        mkpath = '/local/home/yanzhang25/tmp'
    elif 'emerald' in hostname:
        mkpath = '/home/yzhang/tmp'
    print(mkpath)
    exit()
    return mkpath


hostname = socket.gethostname()



















