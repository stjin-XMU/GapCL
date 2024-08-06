from .network import *

def build_model(model_name, args):
    if model_name == 'GCN':
        model = GCN(args)
    return model