import torch
import numpy as np
import pandas as pd


def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f=[]
    for size in data.size.numpy()*3:
        f.append(np.abs(forces[start:start+size].numpy()).max())
        start += size
    return f


def infer(loader, net):
    f_actual = []
    e_actual = []
    e_pred = []
    f_pred = []
    size = []

    for data in loader:

        # extract ground truth values
        e_actual += data.y.tolist()
        f_actual += get_max_forces(data, data.f)
        size += data.size.tolist()
        # get predicted values
        data.pos.requires_grad = True
        e = net(data)
        
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        e_pred += e.tolist()
        f_pred += get_max_forces(data, f)
        

    # return as dataframe
    return pd.DataFrame({'e_actual': e_actual, 'f_actual': f_actual, 'e_pred': e_pred, 'f_pred': f_pred, 'cluster_size': size})

