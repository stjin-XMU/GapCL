from rdkit import Chem
from .utils import onek_encoding_unk

BOND_TYPE = [Chem.rdchem.BondType.SINGLE,Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,Chem.rdchem.BondType.AROMATIC]

class BondFeaturizer(object):
    def __init__(self, name, dim, descrete=False):
        self.name = name
        self.dim = dim
        self.descrete = descrete

    def featurize(self, bond: Chem.rdchem.Bond):
        raise NotImplementedError


class BF1(BondFeaturizer):
    def __init__(self):
        super(BF1, self).__init__('set1', 14)

    def featurize(self, bond: Chem.rdchem.Bond):
        if bond is None:
            fbond = [1] + [0] * (self.dim - 1)  # 1 for bond in None
        else:
            bt = bond.GetBondType()
            fbond = [
                *onek_encoding_unk(bt,BOND_TYPE),
                (bond.GetIsConjugated() if bt is not None else 0),
                (bond.IsInRing() if bt is not None else 0)
            ]
            fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
        return fbond


class BF2(BondFeaturizer):
    def __init__(self):
        super(BF2, self).__init__('set2', 5, True)

    def featurize(self, bond: Chem.rdchem.Bond):
        bt = bond.GetBondType()
        fbond = [BOND_TYPE.index(bt)+1]
        return fbond