# debug_tox21.py
import deepchem as dc
import numpy as np
from rdkit import Chem
import pandas as pd

# Define the featurizer
featurizer = dc.feat.CircularFingerprint(size=1024)

# Load raw Tox21 CSV file
print("Loading raw Tox21 dataset...")
dataset_file = dc.utils.get_data_dir() + "/tox21.csv.gz"  # Default path DeepChem uses
df = pd.read_csv(dataset_file)

# Debug featurization
print("Debugging featurization...")
features = []
for i, smi in enumerate(df['smiles'][:100]):  # Limit to first 100 for quick debugging
    mol = Chem.MolFromSmiles(smi)
    feat = featurizer.featurize([smi])[0] if mol else None
    if feat is not None and isinstance(feat, np.ndarray) and feat.shape == (1024,):
        print(f"SMILES {i}: {smi}, Feature shape: {feat.shape}")
    else:
        print(f"SMILES {i}: {smi}, Issue: Feature={feat if feat is not None else 'None'}, Type={type(feat)}, Shape={feat.shape if isinstance(feat, np.ndarray) else 'N/A'}")
    features.append(feat)

print("Attempting to convert features to array...")
try:
    feature_array = np.asarray(features)
    print(f"Feature array shape: {feature_array.shape}")
except ValueError as e:
    print(f"Error converting to array: {e}")