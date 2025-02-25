# utils.py
import deepchem as dc
import numpy as np
from rdkit import Chem

def load_protein_smiles(smiles):
    """Convert SMILES string to a DeepChem-compatible format."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    featurizer = dc.feat.CircularFingerprint(size=1024)
    features = featurizer.featurize([smiles])
    dataset = dc.data.NumpyDataset(X=features, ids=[smiles])
    return dataset

def format_results(results_dict):
    """Format and print results in a readable way."""
    for category, predictions in results_dict.items():
        print(f"\n=== {category} ===")
        for key, value in predictions.items():
            print(f"{key}: {value}")