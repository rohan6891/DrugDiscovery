# molecule_generation.py
import deepchem as dc
import numpy as np
from deepchem.models import BasicMolGANModel
from deepchem.feat import MolGanFeaturizer
from deepchem.data import NumpyDataset
from deepchem.splits import RandomSplitter  # Import splitter for new API
import os
from rdkit import Chem

def generate_target_specific_molecules(protein_pdb_path, num_molecules=10):
    """Generate molecules targeting the protein's binding pocket using MolGAN."""
    # Use MolGanFeaturizer for molecular graphs
    featurizer = MolGanFeaturizer()
    
    # Load a dataset of molecules (e.g., ZINC15) for training MolGAN
    # Replace 'split' with 'splitter' to address deprecation warning
    tasks, datasets, transformers = dc.molnet.load_zinc15(
        featurizer='ECFP',
        splitter=RandomSplitter()  # Use RandomSplitter instead of split='random'
    )
    train_dataset, valid_dataset, test_dataset = datasets
    
    # Convert DiskDataset to NumpyDataset by manually collecting data
    X_list, y_list, w_list, ids_list = [], [], [], []
    batch_size = 100  # Adjust batch size based on memory constraints
    for X_batch, y_batch, w_batch, ids in train_dataset.iterbatches(batch_size=batch_size):
        X_list.append(X_batch)
        y_list.append(y_batch)
        w_list.append(w_batch)
        ids_list.append(ids)
    X = np.concatenate(X_list, axis=0) if X_list else np.array([])
    y = np.concatenate(y_list, axis=0) if y_list else np.array([])
    w = np.concatenate(w_list, axis=0) if w_list else np.array([])
    ids = np.concatenate(ids_list, axis=0) if ids_list else np.array([])
    train_dataset = NumpyDataset(X=X, y=y, w=w, ids=ids)
    
    # Initialize MolGAN for molecule generation with explicit PyTorch backend
    model = BasicMolGANModel(
        n_tasks=1,  # Single task (binding score)
        n_features=1024,
        layer_sizes=[64, 64],
        dropout=0.2,
        model_dir="molgan_model",
        backend='pytorch'  # Explicitly set PyTorch backend
    )
    
    # Train or load pre-trained model (simplified; use actual training for real results)
    if not os.path.exists("molgan_model/checkpoint"):
        # Use iterbatches to provide batches to fit_gan if direct call fails
        try:
            print("Attempting to fit GAN with full dataset...")
            model.fit_gan(train_dataset)  # Let fit_gan handle batching internally
        except TypeError as e:
            print(f"Full dataset fit failed with: {e}")
            print("Falling back to batch-wise training...")
            batch_size = 32  # Typical batch size for GAN training
            for X_batch, y_batch, w_batch, ids in train_dataset.iterbatches(batch_size=batch_size):
                print("In the Loop for training")
                print("X_batch shape:", X_batch.shape)
                print("y_batch shape:", y_batch.shape)
                print("w_batch shape:", w_batch.shape)
                print("ids shape:", ids.shape)
                batch_dataset = NumpyDataset(X=X_batch, y=y_batch, w=w_batch, ids=ids)
                model.fit_gan(batch_dataset)  # Pass individual NumpyDataset for each batch
                break  # Remove this break for full training; kept for quick test
    
    # Generate molecules
    generated_mols = model.generate(num_molecules, max_atoms=50)
    generated_smiles = [mol.to_smiles() for mol in generated_mols if mol is not None]
    
    return generated_smiles

def dock_molecule(smiles, protein_pdb_path, pocket):
    """Dock a single molecule against the protein pocket using Vina."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string for docking: {smiles}")
    
    import os
    import subprocess
    import tempfile
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Convert ligand SMILES to SDF
        ligand_sdf = os.path.join(temp_dir, "temp_ligand.sdf")
        Chem.MolToMolFile(mol, ligand_sdf)
        
        # Convert SDF to PDBQT (Vina needs PDBQT)
        ligand_pdbqt = os.path.join(temp_dir, "temp_ligand.pdbqt")
        subprocess.run(['obabel', ligand_sdf, '-O', ligand_pdbqt, '-xh'], check=True)
        
        # Convert protein PDB to PDBQT
        protein_pdbqt = os.path.join(temp_dir, "temp_protein.pdbqt")
        subprocess.run(['obabel', protein_pdb_path, '-O', protein_pdbqt, '-xr', '-h'], check=True, capture_output=True, text=True)
        
        # Prepare Vina output
        output_pdbqt = os.path.join(temp_dir, "output.pdbqt")
        
        # Run Vina docking
        centroid = pocket.center()  # Get pocket centroid
        box_size = [20, 20, 20]  # Default box size in Å
        vina_cmd = [
            '/usr/bin/vina',
            '--receptor', protein_pdbqt,
            '--ligand', ligand_pdbqt,
            '--out', output_pdbqt,
            '--center_x', str(centroid[0]),
            '--center_y', str(centroid[1]),
            '--center_z', str(centroid[2]),
            '--size_x', str(box_size[0]),
            '--size_y', str(box_size[1]),
            '--size_z', str(box_size[2]),
            '--exhaustiveness', '8',
            '--num_modes', '1'
        ]
        try:
            result = subprocess.run(vina_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print("Vina failed for SMILES:", smiles)
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise
        
        # Extract score from Vina output (Vina 1.2.5 table format)
        output_lines = result.stdout.splitlines()
        score = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith("1"):  # Look for mode 1 line in table
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "1":
                    score = float(parts[1])  # Affinity score in second column
                    break
        if score is None:
            print("Vina output for SMILES:", smiles, "\n", result.stdout)
            raise ValueError(f"Failed to extract docking score for SMILES: {smiles}")
        
        return score