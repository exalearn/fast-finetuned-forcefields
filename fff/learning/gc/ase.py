"""ASE interface to the model"""
from io import BytesIO

import torch
from ase.calculators.calculator import Calculator, all_changes

from fff.learning.gc.functions import eval_batch
from fff.learning.gc.models import SchNet
from fff.learning.gc.data import convert_atoms_to_pyg


class SchnetCalculator(Calculator):
    """ASE interface to trained model

    Args:
        model: SchNet
    """

    implemented_properties = ['forces', 'energy']
    default_parameters = {'device': 'cpu'}
    nolabel = True

    def __init__(self, model: SchNet, device: str = 'cpu', **kwargs):
        Calculator.__init__(self, **kwargs)
        self.net = model
        self.parameters['device'] = device
        self.net.to(device)

    def __getstate__(self):
        # Remove the model from the serialization dictionary
        state = self.__dict__.copy()
        net = state.pop('net')

        # Serialize it using Torch's "save" functionality
        fp = BytesIO()
        torch.save(net, fp)
        state['_net'] = fp.getvalue()
        return state

    def __setstate__(self, state):
        # Load it back in
        fp = BytesIO(state.pop('_net'))
        state['net'] = torch.load(fp, map_location='cpu')
        state['net'].to(state['parameters']['device'])
        self.__dict__ = state

    def to(self, device: str):
        """Move the model to a certain device"""
        self.parameters['device'] = device
        self.net.to(device)

    @property
    def device(self):
        return self.parameters['device']

    def calculate(
            self, atoms=None, properties=None, system_changes=all_changes, boxsize=None,
    ):
        if properties is None:
            properties = self.implemented_properties

        if atoms is not None:
            self.atoms = atoms.copy()

        Calculator.calculate(self, atoms, properties, system_changes, boxsize=None)

        # Convert the atoms object to a PyG Data
        data = convert_atoms_to_pyg(atoms)
        data.batch = torch.zeros((len(atoms)))  # Fake a single-item batch

        # Run the "batch"
        data.to(self.device)
        energy, gradients = eval_batch(self.net, data)
        self.results['energy'] = energy.item()
        self.results['forces'] = gradients.cpu().detach().numpy().squeeze()

        
class PBCSchnetCalculator(Calculator):
    """ASE interface to trained model
    """
    implemented_properties = ['forces', 'energy']
    nolabel = True

    def __init__(self, best_model, atoms=None, boxsize=None, **kwargs):
        Calculator.__init__(self, **kwargs)
        
        state=torch.load(best_model)
    
        # remove module. from statedict keys (artifact of parallel gpu training)
        state = {k.replace('module.',''):v for k,v in state.items()}
        num_gaussians = state[f'basis_expansion.offset'].shape[0]
            
        num_filters = state[f'interactions.0.mlp.0.weight'].shape[0]
        num_interactions = len([key for key in state.keys() if '.lin.bias' in key])

        net = SchNet(num_features = num_filters,
                     num_interactions = num_interactions,
                     num_gaussians = num_gaussians,
                     cutoff = 6.0,
                     boxsize = boxsize,
                    )

        net.load_state_dict(state)

        self.net = net
        self.atoms = atoms
        self.atoms.set_cell(box_size)

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        if atoms is not None:
            self.atoms = atoms.copy()
        
        Calculator.calculate(self, atoms, properties, system_changes)

        energy, gradients = schnet_eg(self.atoms, self.net)
        self.results['energy'] = energy
        self.results['forces'] = -gradients
        
def schnet_eg(atoms, net, device='cpu'):
    """
    Takes in ASE atoms and loaded net and predicts energy and gradients
    args: atoms (ASE atoms object), net (loaded trained Schnet model)
    return: predicted energy (eV), predicted gradients (eV/angstrom)
    """
    types = {'H': 0, 'O': 1}
    atom_types = [1, 8]
    
    #center = False
    #if center:
    #    pos = atoms.get_positions() - atoms.get_center_of_mass()
    #else:
    
    pos = atoms.get_positions()
    pos = torch.tensor(pos, dtype=torch.float) #coordinates
    size = int(pos.size(dim=0)/3)
    type_idx = [types.get(i) for i in atoms.get_chemical_symbols()]
    atomic_number = atoms.get_atomic_numbers()
    z = torch.tensor(atomic_number, dtype=torch.long)
    x = F.one_hot(torch.tensor(type_idx, dtype=torch.long),
                              num_classes=len(atom_types))
    data = Data(x=x, z=z, pos=pos, size=size, batch=torch.tensor(np.zeros(size*3), dtype=torch.int64), idx=1)

    data = data.to(device)
    data.pos.requires_grad = True
    if self.atoms.get_pbc():
        e = net(data, boxsize=self.atoms.get_cell())
    else:
        e = net(data)
    f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=True)[0].cpu().data.numpy()
    e = e.cpu().data.numpy()

    return e.item()/23.06035, f/23.06035