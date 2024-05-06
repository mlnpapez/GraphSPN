from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import MolsToGridImage, rdMolDraw2D # , MolsMatrixToGridImage

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

def highlight_grid(smiles_mat, smarts_patts, path="mol.png"):
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
            mol = Chem.MolFromSmiles(smile_mol)
            AllChem.GenerateDepictionMatching2DStructure(mol, patt)
            mols.append(mol)
            hit_at, hit_bond = get_hit(mol, patt)
            hit_ats.append(hit_at)
            hit_bonds.append(hit_bond)

    dopts = rdMolDraw2D.MolDrawOptions()
    dopts.setHighlightColour((0,.9,.9,.8))
    dopts.highlightBondWidthMultiplier = 15
    img = MolsToGridImage(mols, highlightAtomLists=hit_ats, highlightBondLists=hit_bonds, 
                        molsPerRow=molsPerRow, subImgSize=(400,400), drawOptions=dopts)
    img.save(path)


if __name__ == "__main__":
    slist = ['CC1=CC2=C(C=C1)C(=CN2CCN1CCOCC1)C(=O)C1=CC=CC2=C1C=CC=C2',
        'CCCCCN1C=C(C2=CC=CC=C21)C(=O)C3=CC=CC4=CC=CC=C43',
        'CC1COCCN1CCN1C=C(C(=O)C2=CC=CC3=C2C=CC=C3)C2=C1C=CC=C2',
        'CC1=CC=C(C(=O)C2=CN(CCN3CCOCC3)C3=C2C=CC=C3)C2=C1C=CC=C2',
        'CC1=C(CCN2CCOCC2)C2=C(C=CC=C2)N1C(=O)C1=CC=CC2=CC=CC=C12',
        'CN1CCN(C(C1)CN2C=C(C3=CC=CC=C32)C(=O)C4=CC=CC5=CC=CC=C54)C']
    smile_mols = [slist, slist]
    smart_patts = ['c1c2ccccc2ccc1', 'C=O']

    highlight_grid(smile_mols, smart_patts)
