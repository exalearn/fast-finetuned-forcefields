import os
import os.path as op
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
from utils.water_dataset import PrepackedDataset, gc_WaterDataSet

import sys

def dataloader(datasets,
               mode,
               split = '00', 
               batch = 256,
               shuffle=True,
               splitdir = './data/splits/',
               datadir = './data/cached_dataset/'
               ):
    """
    dataset = str or list of str for each dataset label ['min', 'nonmin']
    mode = ['train','val','examine','test']
    """

    if isinstance(datasets, list):
        data = []
        for ds in datasets:
            dataset = PrepackedDataset(None, 
                                       op.join(splitdir,f'split_{split}_{ds}.npz'), 
                                       ds, 
                                       directory=datadir)
            data.append(dataset.load_data(mode))
            
        loader = DataLoader(ConcatDataset(data), batch_size=batch, shuffle=shuffle)
        
    else:
        dataset = PrepackedDataset(None, 
                                   op.join(splitdir,f'split_{split}_{datasets}.npz'), 
                                   datasets, 
                                   directory=datadir)
        data = dataset.load_data(mode)

        print(f'{datasets} data loaded from {datadir}')

        batch_size = batch if len(data) > batch else len(data)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader

#TODO all of batch_dataloader (single, train val, multiple examines
def batch_dataloader(args, datasets,
               modes=['train','val','examine'],
               split = '00', 
               batch = 256,
               shuffle=True,
               splitdir = './data/splits/',
               datadir = './data/cached_dataset/'
               ):
    """
    dataset = str or list of str for each dataset label ['min', 'nonmin']
    mode = ['train','val','examine','test']
    """

    if isinstance(args.datasets, list):
        data = []
        for ds in args.datasets:
            dataset = PrepackedDataset(None, 
                                       op.join(args.savedir,f'split_{split}_{ds}.npz'), 
                                       ds, 
                                       directory=datadir)
            
            data.append(dataset.load_data(mode))
            
        loader = DataLoader(ConcatDataset(data), batch_size=batch, shuffle=shuffle)
        
    else:
        dataset = PrepackedDataset(None, 
                                   op.join(args.savedir,f'split_{split}_{args.datasets}.npz'), 
                                   datasets, 
                                   directory=datadir)
        data = dataset.load_data(mode)

        batch_size = batch if len(data) > batch else len(data)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

    return loader


def test_dataloader(model_cat, 
                    dataset,
                    batch = 256,
                    splitdir = './data/splits/',
                    datadir = './data/cached_dataset/'
                    ):

    if model_cat == 'ipu':
        sys.path.insert(0, '/qfs/projects/ecp_exalearn/designs/NN_TBPMC/IPU_trained_models/pnnl_sandbox/schnet')
        import transforms
        
        # transforms for IPU model
        transform = transforms.create_transform(mode = "knn", k = 28,
                                                cutoff = 6.0, use_padding = False,
                                                max_num_atoms = 75, max_num_edges = None,
                                                use_qm9_energy = False, use_standardized_energy = False)
        data = gc_WaterDataSet(sample=f'split_00_test_set_{dataset}', 
                               root='./',
                               split_file = op.join(splitdir,f'split_00_{dataset}.npz'),
                               h5py_file = op.join(datadir,f'{dataset}_data.hdf5'),
                               idx_type = 'test',
                               pre_transform=transform)
    else:
        dataset = PrepackedDataset(None, op.join(splitdir,f'split_00_{dataset}.npz'), dataset, 
                                   directory=datadir)
        data = dataset.load_data('test')
    
    print('done preprocessing')

    batch_size = batch if len(data) > batch else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader


def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f_actual=[]
    for size in data.size.numpy()*3:
        f_actual.append(np.abs(data.f[start:start+size].numpy()).max())   
        start += size
    return f_actual
