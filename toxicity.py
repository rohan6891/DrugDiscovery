# toxicity.py (updated with CircularFingerprint(size=240))
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
import tensorflow as tf
import torch  # Ensure PyTorch is imported

def predict_toxicity(smiles_dataset):
    """Predict toxicity properties using a PyTorch model."""
    featurizer = dc.feat.CircularFingerprint(size=240)  # Reduced to 240 features
    
    print("Loading raw Tox21 dataset...")
    dataset_file = dc.utils.get_data_dir() + "/tox21.csv.gz"
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found. Downloading...")
        dc.molnet.load_tox21()
        if not os.path.exists(dataset_file):
            raise FileNotFoundError(f"Failed to download {dataset_file}")
    
    df = pd.read_csv(dataset_file)
    
    tox21_tasks = ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
                   'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        
    # toxicity.py (skip IRVTransformer)
    X, y, w, ids = [], [], [], []
    for i, row in df.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        feat = featurizer.featurize([smi])[0] if mol else None
        if mol and feat is not None and isinstance(feat, np.ndarray) and feat.shape == (240,):
            X.append(feat)
        else:
            print(f"Skipping inconsistent feature for {smi} or invalid molecule, shape: {feat.shape if feat is not None else 'None'}")
            X.append(np.zeros(240))  # Use zeros for invalid features, ensuring consistent shape of 240
        labels = row[tox21_tasks].values.astype(float)
        labels = np.nan_to_num(labels, nan=0.0)
        y.append(labels)
        w.append(np.ones_like(labels))
        ids.append(smi)

    # Convert and print final X shape
    X = np.array(X, dtype=np.float32)
    print(f"Final X shape: {X.shape}")  # Should print (7831, 240)
    y = np.array(y, dtype=np.float32)
    w = np.array(w, dtype=np.float32)
    ids = np.array(ids)

    train_dataset = dc.data.NumpyDataset(X=X, y=y, w=w, ids=ids)
    print(f"Processed dataset size: {len(train_dataset)}")

    n_tasks = len(tox21_tasks)

    model = dc.models.MultitaskIRVClassifier(
        n_tasks=n_tasks,
        n_features=1024,  # Match CircularFingerprint size
        layer_sizes=[1000, 500],
        dropouts=0.2,
        model_dir="tox21_model",
        use_irv=True,  # Keep use_irv=True if model expects IRV, but skip transformer
        backend='pytorch'
    )

    # Manually set _input_dtypes and _input_shapes (adjust for PyTorch if needed)
    dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32}
    model._input_dtypes = [dtype_map.get(t.dtype if hasattr(t, 'dtype') else t, np.float32) for t in model.model.inputs]
    model._input_shapes = [(None, 240)]  # Match the featurizer size
    model._inputs_built = True  # Skip _create_inputs if needed

    print("Model inputs before fit:", [inp.dtype for inp in model.model.inputs])
    print("Raw model inputs:", model.model.inputs)
    print("Set _input_dtypes:", model._input_dtypes)
    print("Set _input_shapes:", model._input_shapes)

    # Use fit method for PyTorch
    model.fit(train_dataset, nb_epoch=10)

    # Fix prediction
    X_pred = smiles_dataset.X
    y_pred = np.zeros((1, n_tasks), dtype=np.float32)  # Dummy y with 12 tasks
    w_pred = np.ones((1, n_tasks), dtype=np.float32)
    pred_dataset = dc.data.NumpyDataset(X=X_pred, y=y_pred, w=w_pred, ids=smiles_dataset.ids)
    predictions = model.predict(pred_dataset)[0]

    return {
        "Toxicity Prediction (Tox21 Example)": predictions[0],
        "Carcinogenicity": "Not Implemented",
        "Mutagenicity (Ames Test)": "Not Implemented",
        "Hepatotoxicity": "Not Implemented",
        "hERG Inhibition": "Not Implemented"
    }