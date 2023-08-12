from rdkit.Chem import Draw
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import torch
import os
import json
import pdb

            
def mol2array(mol):
    img = Draw.MolToImage(mol, kekulize=False)
    array = np.array(img)[:, :, 0:3]
    return array

def check(smile):
    smile = smile.split('.')
    smile.sort(key = len)
    try:
        mol = Chem.MolFromSmiles(smile[-1], sanitize=False)
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        return True
    except Exception:
        return False

def mol2file(m, name):
    AllChem.Compute2DCoords(m)
    img = Draw.MolToImage(m)
    Draw.MolToFile(m, os.path.join('./img', name))


class MolecularGraph:
    def __init__(self, num_atoms):
        self.num_atoms = num_atoms
        self.adj_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.int)
        self.atom_elements = [None] * num_atoms
        self.bond_types = [[''] * num_atoms for _ in range(num_atoms)]

    def add_atom(self, index, element):
        self.atom_elements[index] = element

    def add_bond(self, atom1, atom2, bond_type):
        if bond_type == 'single' or bond_type == 'aromatic':
            self.adj_matrix[atom1, atom2] = 1
            self.adj_matrix[atom2, atom1] = 1
        elif bond_type == 'double':
            self.adj_matrix[atom1, atom2] = 2
            self.adj_matrix[atom2, atom1] = 2
        elif bond_type == 'triple':
            self.adj_matrix[atom1, atom2] = 3
            self.adj_matrix[atom2, atom1] = 3
        self.bond_types[atom1][atom2] = bond_type
        self.bond_types[atom2][atom1] = bond_type

def result2mol(args): # for threading
    element, mask, bond, aroma, charge, reactant = args
    # [L], [L], [L, 4], [l], [l]
    try:
        if isinstance(mask, torch.Tensor):
            mask = mask.ne(1)
        else:
            mask = mask != 1
    except AttributeError:
        raise AttributeError('mask should be a tensor or numpy array')

    l = element.shape[0]

    mol = Chem.RWMol() # 创建一个分子

    if isinstance(element, torch.Tensor):
        element = element.cpu().numpy().tolist()
        charge = charge.cpu().numpy().tolist()
        bond = bond.cpu().numpy().tolist()
    else:
        element = element.tolist()
        charge = charge.tolist()
        bond = bond.tolist()

    mol_graph = MolecularGraph(l)

    # add atoms to mol and keep track of index
    node_to_idx = {} # key: index in element; value: index in mol
    for i in range(l):
        if mask[i] == False:
            continue
        a = Chem.Atom(element[i])
        # if not reactant is None and reactant[i]:
        #     a.SetAtomMapNum(i+1)
        # --- 添加了AtomMapping -------------------------------------------------------------------
        a.SetAtomMapNum(i + 1)
        molIdx = mol.AddAtom(a) # returns the index of the newly added atom 返回这个原子是分子中加入的第几个原子
        node_to_idx[i] = molIdx
        mol_graph.add_atom(molIdx, element[i])

    # add bonds between adjacent atoms
    for this in range(l):
        if mask[this] == False:
            continue
        lst = bond[this]
        for j in range(len(bond[0])):
            other = bond[this][j]
            # only traverse half the matrix
            if other >= this or other in lst[0:j] or not this in bond[other]:
                continue
            if lst.count(other)==3 or bond[other].count(this) == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                mol_graph.add_bond(this, other, 'triple')
            elif lst.count(other) == 2 or bond[other].count(this) == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                mol_graph.add_bond(this, other, 'double')
            else:
                if aroma[this]==aroma[other] and aroma[this]>0: 
                    bond_type = Chem.rdchem.BondType.AROMATIC
                    mol_graph.add_bond(this, other, 'aromatic')
                else:
                    bond_type = Chem.rdchem.BondType.SINGLE
                    mol_graph.add_bond(this, other, 'single')
                mol.AddBond(node_to_idx[this], node_to_idx[other], bond_type)
                 
    for i, item in enumerate(charge):
        if mask[i] == False:
            continue
        if not item == 0:
            atom = mol.GetAtomWithIdx(node_to_idx[i])
            atom.SetFormalCharge(item)

    # Convert RWMol to Mol object
    mol = mol.GetMol() 
    Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS) # 卫生化过程可能会对分子进行一些改变，以满足化学规则和标准。
    smile = Chem.MolToSmiles(mol)

    # print('smile','*'*20)
    # print(smile)

    return mol, smile, check(smile), mol_graph
    # return mol, smile, check(smile)


def visualize(element, mask, bond, aroma, charge, reactant=None):
    mol, smile, _, mol_graph = result2mol((element, mask, bond, aroma, charge, reactant))
    # mol, smile, _ = result2mol((element, mask, bond, aroma, charge, reactant))
    array = mol2array(mol)
    return array, smile


def get_adjgraph(element, mask, bond, aroma, charge, reactant=None):
    mol, smile, _, mol_graph = result2mol((element, mask, bond, aroma, charge, reactant))
    # mol, smile, _ = result2mol((element, mask, bond, aroma, charge, reactant))
    return mol_graph

def get_diff_adj(element, mask, bond, aroma, charge,
                  bond1, aroma1, charge1,
                 reactant=None, reactant1=None):
    mol, smile, _, mol_graph = result2mol((element, mask, bond, aroma, charge, reactant))
    mol1, smile1, _, mol_graph1 = result2mol((element, mask, bond1, aroma1, charge1, reactant1))

    adj = mol_graph.adj_matrix
    adj1 = mol_graph1.adj_matrix

    assert adj.shape == adj1.shape
    diff = adj - adj1
    return diff, adj, adj1
