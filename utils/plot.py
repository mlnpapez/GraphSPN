import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D
from utils.molecular import isvalid

def marginalize(network, nd_nodes, num_empty, num_full):
    with torch.no_grad():
        if num_empty > 0:
            mx = torch.zeros(nd_nodes,           dtype=torch.bool)
            ma = torch.zeros(nd_nodes, nd_nodes, dtype=torch.bool)
            mx[num_full:   ] = True
            ma[num_full:, :] = True
            ma[:, num_full:] = True
            m = torch.cat((mx.unsqueeze(1), ma), dim=1)
            marginalization_idx = torch.arange(nd_nodes+nd_nodes**2)[m.view(-1)]

            network.set_marginalization_idx(marginalization_idx)
        else:
            network.set_marginalization_idx(None)

def get_hit(mol, patt):
    """Function returns atoms/vertices to highlight and bonds/edges to highlight
        in molecule `mol` given the pattern.
    Parameters:
        mol (rdkit.Chem.rdchem.Mol): Molecule to plot
        patt (rdkit.Chem.rdchem.Mol): Pattern to highlight in `mol`
    Returns:
        hit_ats, hit_bonds ( tuple[list, list]): atoms/vertices to highlight and bonds/edges to highlight
    """
    hit_ats = list(mol.GetSubstructMatch(patt))
    hit_bonds = []

    for bond in patt.GetBonds():
       aid1 = hit_ats[bond.GetBeginAtomIdx()]
       aid2 = hit_ats[bond.GetEndAtomIdx()]
       hit_bonds.append(mol.GetBondBetweenAtoms(aid1,aid2).GetIdx())

    return hit_ats, hit_bonds

def highlight_grid(smiles_mat, smarts_patts, fname="cond_mols", useSVG=False):
    """Function plots aligned grid of molecules with highlited patterns.
    Parameters:
        mol (list[list[str]]): SMILE molecule to plot
        patt (list[str]): SMILE/SMART patterns to highlight in `mol`
    Returns:
        hit_ats, hit_bonds (tuple[list, list]): atoms/vertices to highlight and bonds/edges to highlight
    """
    molsPerRow = len(smiles_mat[0])

    mols = []
    hit_ats = []
    hit_bonds = []

    for row, smile_patt in zip(smiles_mat, smarts_patts):
        patt = Chem.MolFromSmarts(smile_patt)
        AllChem.Compute2DCoords(patt)
        for smile_mol in row:
            mol = Chem.MolFromSmiles(smile_mol, sanitize=False)
            AllChem.GenerateDepictionMatching2DStructure(mol, patt, acceptFailure=True)
            mols.append(mol)
            hit_at, hit_bond = get_hit(mol, patt)
            # hit_at, hit_bond = [], []
            hit_ats.append(hit_at)
            hit_bonds.append(hit_bond)

    dopts = rdMolDraw2D.MolDrawOptions()
    dopts.setHighlightColour((.0,.8,.9))
    dopts.highlightBondWidthMultiplier = 15
    img = MolsToGridImage(mols, highlightAtomLists=hit_ats, highlightBondLists=hit_bonds, 
                        molsPerRow=molsPerRow, subImgSize=(400,400), drawOptions=dopts, useSVG=useSVG)
    if useSVG:
         with open(f'{fname}.svg', 'w') as f:
            img = img.replace("<rect style='opacity:1.0", "<rect style='opacity: 0")  # for transparent background
            f.write(img)
    else:    
        img.save(f'{fname}.png')

def joint_grid(model, nrows, ncols, useSVG=False):
    mols, smls = model.sample(10*nrows*ncols)
    valid_mols = [mol for mol in mols if isvalid(mol)]
    img = MolsToGridImage(valid_mols[:nrows*ncols], molsPerRow=ncols, subImgSize=(400, 400), useSVG=useSVG)
    if useSVG:
         with open('joint_mols.svg', 'w') as f:
            img = img.replace("<rect style='opacity:1.0", "<rect style='opacity: 0")  # for transparent background
            f.write(img)
    else:    
        img.save('joint_mols.png')

if __name__ == "__main__":
    # slist = ['CC1=CC2=C(C=C1)C(=CN2CCN1CCOCC1)C(=O)C1=CC=CC2=C1C=CC=C2',
    #     'CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43',
    #     'CC1COCCN1CCN1C=C(C(=O)C2=CC=CC3=C2C=CC=C3)C2=C1C=CC=C2',
    #     'CC1=CC=C(C(=O)C2=CN(CCN3CCOCC3)C3=C2C=CC=C3)C2=C1C=CC=C2',
    #     'CC1=C(CCN2CCOCC2)C2=C(C=CC=C2)N1C(=O)C1=CC=CC2=CC=CC=C12',
    #     'CN1CCN(C(C1)CN2C=C(C3=CC=CC=C32)C(=O)C4=CC=CC5=CC=CC=C54)C']
    # smile_mols = [slist, slist]
    # smart_patts = ['c1c2ccccc2ccc1', 'C=O']

    # highlight_grid(smile_mols, smart_patts)

    slist = ['OCCC1C2=CC[O+]1C2', 'OC1=CNC2=C1COC2', 'CC1C2=CC[O+]1C2', 'C1=C2C[O+](C1)C2', 'OC1=NOC2=C1COC2']
    smls = [slist]
    patts_smls = ['C1OCC=C1']
    highlight_grid(smls, patts_smls)

