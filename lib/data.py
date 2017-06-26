import sys, os, pickle, settings
import numpy as np


def load_pk(f_name):
    with open(f_name, 'rb') as f:
        pk = pickle.load(f)

    return pk

def save_pk(f_name, data):
    with open(f_name, 'wb') as f:
        pickle.dump(data, f)

    print('Data saved to file: {}\n'.format(f_name))

def save_log(f_name, text):
    with open(f_name, 'w') as f:
        f.write(text)

    print('Log saved to file: {}\n'.format(f_name))

def read_txt(f_name, delimiter='\n', trim=-1):
    with open(f_name, 'r') as f:
        return f.read().split(delimiter)[:trim]
