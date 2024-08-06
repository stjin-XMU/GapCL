
from rdkit import Chem
from .utils import onek_encoding_unk

ELEMENTS = [1, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 22, 24, 25, 26, 29, 30, 33, 34, 35, 43, 53, 78, 79, 80, 81, 83]

ATOM_FEATURES = {
    'atomic_num': ELEMENTS,
    'degree': [0, 1, 2, 3, 4, 5, 6],
    'formal_charge': [-1, -2, 1, 2, 3, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [1, 2, 3, 4, 5, 6, 7],
}

class AtomFeaturizer(object):
    def __init__(self, name, dim, descrete=False):
        self.name = name
        self.dim = dim
        self.descrete = descrete

    def featurize(self, bond: Chem.rdchem.Bond):
        raise NotImplementedError


class AF1(AtomFeaturizer):
    def __init__(self):
        super(AF1, self).__init__('set1', 64)

    def featurize(self, atom: Chem.rdchem.Atom):
        features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0] + \
            [atom.GetMass() * 0.01]  # scaled to about the same range as other features
        return features

'''for atomic num only'''

class AF2(AtomFeaturizer):
    def __init__(self):
        super(AF2, self).__init__('set2', 28)

    def featurize(self, atom: Chem.rdchem.Atom):
        features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num'])
        return features
    

class AF4(AtomFeaturizer):
    def __init__(self):
        super(AF4, self).__init__('set4', 56)

    def featurize(self, atom: Chem.rdchem.Atom):
        features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) * 2
        return features
    

class AF3(AtomFeaturizer):
    def __init__(self):
        super(AF3, self).__init__('set1', 63)

    def featurize(self, atom: Chem.rdchem.Atom):
        features = onek_encoding_unk(atom.GetAtomicNum(), ATOM_FEATURES['atomic_num']) + \
            onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
            onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
            onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
            onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
            onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
            [1 if atom.GetIsAromatic() else 0]
        return features