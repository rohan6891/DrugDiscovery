# main.py
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
from utils import load_protein_smiles, format_results
from drug_likeness import compute_drug_likeness
from toxicity import predict_toxicity
from pharmacokinetics import predict_admet
from bioactivity import predict_bioactivity
from synthesis import predict_synthesis

def run_pipeline(smiles, protein_pdb_path):
    """Run the full drug discovery pipeline with TensorFlow."""
    smiles_dataset = load_protein_smiles(smiles)
    
    results = {
        "Drug-Likeness & Physicochemical Properties": compute_drug_likeness(smiles_dataset),
        "Toxicity & Safety": predict_toxicity(smiles_dataset),
        "Pharmacokinetics (ADMET)": predict_admet(smiles_dataset),
        "Bioactivity & Efficacy": predict_bioactivity(smiles_dataset, protein_pdb_path),
        "Synthesis & Stability": predict_synthesis(smiles_dataset)
    }
    
    format_results(results)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python main.py <SMILES> <protein_pdb_path>")
        sys.exit(1)
    
    smiles = sys.argv[1]
    protein_pdb_path = sys.argv[2]
    run_pipeline(smiles, protein_pdb_path)