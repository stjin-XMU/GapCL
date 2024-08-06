from .atom import *
from .bond import *

def get_featurizer(featurizer):
    af_set, bf_set = featurizer
    if af_set == 'set1':
        af = AF1()
    elif af_set == 'set2':
        af = AF2()
    elif af_set == 'set3':
        af = AF3()
    else:
        raise ValueError(f'atom feature `{af_set}` set not implemented')
    
    if bf_set == 'set1':
        bf = BF1()
    elif bf_set == 'set2':
        bf = BF2()
    else:
        raise ValueError(f'bond feature set `{bf_set}` not implemented')
    
    return af,bf

    