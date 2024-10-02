from .network import *

def build_model(model_name, args):
    if model_name == 'MPNN':
        model = MPNN(args)
    return model
