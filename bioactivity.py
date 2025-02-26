# bioactivity.py
import deepchem as dc
from rdkit import Chem
import os
import subprocess
import tempfile
import molecule_generation  # Import the new file

def predict_bioactivity(smiles_dataset, protein_pdb_path):
    """Predict bioactivity and docking scores, including generated molecules."""
    # Find binding pockets
    finder = dc.dock.binding_pocket.ConvexHullPocketFinder()
    pockets = finder.find_pockets(protein_pdb_path)
    if not pockets:
        raise ValueError("No binding pockets found in protein.")
    
    pocket = pockets[0]  # Use the first pocket
    
    # Dock the input molecule (CCO) as before
    # Validate SMILES
    smiles = smiles_dataset.ids[0]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string for docking.")
    
    # Create temporary files
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
        centroid = pockets[0].center()  # Call the method with ()
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
            print("Vina failed with the following output:")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
            raise
        
        # Extract score from Vina output (Vina 1.2.5 table format)
        output_lines = result.stdout.splitlines()
        score = None
        for i, line in enumerate(output_lines):
            if line.strip().startswith("1"):  # Look for mode 1 line in table
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "1":  # Ensure it's mode 1
                    score = float(parts[1])  # Affinity score in second column
                    break
        if score is None:
            print("Vina output:", result.stdout)
            raise ValueError("Failed to extract docking score from Vina output.")

    # Generate and dock target-specific molecules using molecule_generation
    generated_smiles = molecule_generation.generate_target_specific_molecules(protein_pdb_path, num_molecules=5)
    generated_scores = {}
    for i, gen_smiles in enumerate(generated_smiles):
        try:
            generated_scores[f"generated_molecule_{i+1}"] = molecule_generation.dock_molecule(gen_smiles, protein_pdb_path, pocket)
        except ValueError as e:
            print(f"Failed to dock generated molecule {gen_smiles}: {e}")
            generated_scores[f"generated_molecule_{i+1}"] = None  # Mark as failed

    # Return results
    return {
        "Molecular Docking Score": score,
        "Molecular Docking Scores (Generated)": generated_scores,
        "Target Binding Affinity": "Not Implemented",
        "Enzyme Inhibition/Activation": "Not Implemented",
        "Bioavailability Score": "Not Implemented"
    }