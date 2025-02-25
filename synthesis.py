# synthesis.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
import deepchem as dc
import numpy as np
from rdkit import Chem
from rdkit.Chem import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def predict_synthesis(smiles_dataset):
    """Predict synthesis and stability properties."""
    smiles = smiles_dataset.ids[0]
    mol = Chem.MolFromSmiles(smiles)
    sa_score = sascorer.calculateScore(mol)
    
    tasks, datasets, transformers = dc.molnet.load_delaney(featurizer='ECFP')
    train_dataset, valid_dataset, test_dataset = datasets
    
    n_tasks = len(tasks)  # 1 for Delaney
    model = dc.models.MultitaskRegressor(
        n_tasks=n_tasks,
        n_features=1024,
        layer_sizes=[1000, 500],
        dropouts=0.2,
        model_dir="solubility_model_synthesis"
    )
    
    # Manually set _input_dtypes and _input_shapes
    dtype_map = {'float32': np.float32, 'float64': np.float64, 'int32': np.int32}
    model._input_dtypes = [dtype_map.get(t.dtype if hasattr(t, 'dtype') else t, np.float32) for t in model.model.inputs]
    model._input_shapes = [(None, 1024), (None,)]  # Two inputs
    model._inputs_built = True
    
    print("Model inputs before fit:", [inp.dtype for inp in model.model.inputs])
    print("Raw model inputs:", model.model.inputs)
    print("Set _input_dtypes:", model._input_dtypes)
    print("Set _input_shapes:", model._input_shapes)
    
    model.fit(train_dataset, nb_epoch=10)
    
    # Fix prediction
    X_pred = smiles_dataset.X
    y_pred = np.zeros((1, n_tasks), dtype=np.float32)
    w_pred = np.ones((1, n_tasks), dtype=np.float32)
    pred_dataset = dc.data.NumpyDataset(X=X_pred, y=y_pred, w=w_pred, ids=smiles_dataset.ids)
    solubility_pred = model.predict(pred_dataset)[0][0]
    
    return {
        "Synthetic Accessibility Score": float(sa_score),
        "Chemical Stability": "Not Implemented",
        "Solubility Prediction": float(solubility_pred)
    }