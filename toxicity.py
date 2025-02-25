# toxicity.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import deepchem as dc
import numpy as np
import pandas as pd
from rdkit import Chem
import tensorflow as tf

def predict_toxicity(smiles_dataset):
    """Predict toxicity properties using a TensorFlow model."""
    featurizer = dc.feat.CircularFingerprint(size=1024)
    
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
    
    X, y, w, ids = [], [], [], []
    for i, row in df.iterrows():
        smi = row['smiles']
        mol = Chem.MolFromSmiles(smi)
        feat = featurizer.featurize([smi])[0] if mol else None
        if mol and isinstance(feat, np.ndarray) and feat.shape == (1024,):
            X.append(feat)
        else:
            print(f"Padding inconsistent feature for {smi}")
            X.append(np.zeros(1024))
        labels = row[tox21_tasks].values.astype(float)
        labels = np.nan_to_num(labels, nan=0.0)
        y.append(labels)
        w.append(np.ones_like(labels))
        ids.append(smi)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    w = np.array(w, dtype=np.float32)
    ids = np.array(ids)
    
    train_dataset = dc.data.NumpyDataset(X=X, y=y, w=w, ids=ids)
    print(f"Processed dataset size: {len(train_dataset)}")
    
    n_tasks = len(tox21_tasks)
    model = dc.models.MultitaskClassifier(
        n_tasks=n_tasks,
        n_features=1024,
        layer_sizes=[1000, 500],
        dropouts=0.2,
        model_dir="tox21_model"
    )
    
    # Manually set _input_dtypes and _input_shapes
    dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32}
    model._input_dtypes = [dtype_map.get(t.dtype if hasattr(t, 'dtype') else t, np.float32) for t in model.model.inputs]
    model._input_shapes = [(None, 1024)]  # Shape of your input data
    model._inputs_built = True  # Skip _create_inputs
    
    print("Model inputs before fit:", [inp.dtype for inp in model.model.inputs])
    print("Raw model inputs:", model.model.inputs)
    print("Set _input_dtypes:", model._input_dtypes)
    print("Set _input_shapes:", model._input_shapes)
    
    model.fit(train_dataset, nb_epoch=10)
    
    # Fix prediction by ensuring smiles_dataset has compatible y
    # For prediction, y can be None or a dummy array with shape (1, n_tasks)
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