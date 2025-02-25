# drug_likeness.py
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, MolSurf

def compute_drug_likeness(smiles_dataset):
    """Compute drug-likeness properties."""
    smiles = smiles_dataset.ids[0]
    mol = Chem.MolFromSmiles(smiles)
    
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    tpsa = MolSurf.TPSA(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    
    lipinski_pass = (mw <= 500 and logp <= 5 and h_donors <= 5 and h_acceptors <= 10)
    veber_pass = (rotatable_bonds <= 10 and tpsa <= 140)
    
    return {
        "Molecular Weight (MW)": mw,
        "LogP (Lipophilicity)": logp,
        "Hydrogen Bond Donors": h_donors,
        "Hydrogen Bond Acceptors": h_acceptors,
        "Topological Polar Surface Area (TPSA)": tpsa,
        "Number of Rotatable Bonds": rotatable_bonds,
        "Lipinski's Rule Pass": lipinski_pass,
        "Veber's Rule Pass": veber_pass
    }