# utils.py (updated to match training)
import deepchem as dc
import numpy as np
from rdkit import Chem

def load_protein_smiles(smiles):
    """Convert SMILES string to a DeepChem-compatible format, matching training featurization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")
    featurizer = dc.feat.CircularFingerprint(size=240)  # Match training featurizer
    features = featurizer.featurize([smiles])
    if features is None or features.size == 0:
        raise ValueError(f"No features generated for SMILES: {smiles}")
    X = np.array(features, dtype=np.float32)  # Should be (1, 240)
    y = np.zeros((1, 12), dtype=np.float32)  # Dummy y for 12 Tox21 tasks
    w = np.ones((1, 12), dtype=np.float32)
    ids = np.array([smiles])
    dataset = dc.data.NumpyDataset(X=X, y=y, w=w, ids=ids)
    return dataset

def format_results(results_dict):
    """Format and print results in a readable way."""
    for category, predictions in results_dict.items():
        print(f"\n=== {category} ===")
        for key, value in predictions.items():
            print(f"{key}: {value}")