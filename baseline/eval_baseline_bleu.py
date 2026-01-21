import os
import sys
import argparse
import pickle
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
from transformers import AutoTokenizer

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Add parent directory to path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data_utils import (
    load_id2emb, 
    load_descriptions_from_graphs, 
    PreprocessedGraphDataset, 
    collate_fn
)
from app.evaluate import compute_bleu4
from baseline.train_gcn import MolGNN, DEVICE

# Config
TRAIN_GRAPHS = "data/train_graphs.pkl"
VAL_GRAPHS   = "data/validation_graphs.pkl"
TRAIN_EMB_CSV = "data/train_embeddings.csv"
MODEL_PATH = "model_checkpoint.pt"

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False

def compute_bertscore(predictions, references):
    if not HAS_BERTSCORE:
        return {"BERTScore-F1": -1.0}
    
    P, R, F1 = bert_score(
        predictions,
        references,
        model_type="roberta-base",
        verbose=False
    )
    return {
        "BERTScore-P": P.mean().item(),
        "BERTScore-R": R.mean().item(),
        "BERTScore-F1": F1.mean().item(),
    }

@torch.no_grad()
def evaluate_bleu(model, val_loader, train_embs_tensor, train_descriptions, device):
    """
    1. Encode validation graphs -> val_embs.
    2. Retrieve nearest neighbor from train_embs.
    3. Get description of nearest neighbor.
    4. Compute BLEU and BERTScore against actual validation description.
    """
    model.eval()
    
    # ... (same encoding steps 1 & 2 as before)
    
    # 1. Encode all validation graphs
    val_mol_embs = []
    val_descriptions_gt = [] # Ground truth
    
    print("Encoding validation graphs...")
    for graphs in tqdm(val_loader, desc="Encoding Val"):
        graphs = graphs.to(device)
        mol_emb = model(graphs)
        val_mol_embs.append(mol_emb)
        
        data_list = graphs.to_data_list()
        for data in data_list:
            if hasattr(data, 'description'):
                val_descriptions_gt.append(data.description)
            else:
                val_descriptions_gt.append("")

    val_mol_embs = torch.cat(val_mol_embs, dim=0)
    
    # 2. Similarity Search & Retrieval
    print("Computing similarity matrix...")
    sims = val_mol_embs @ train_embs_tensor.t()
    
    print("Retrieving nearest neighbors...")
    best_indices = sims.argmax(dim=1).cpu().numpy()
    
    predicted_captions = []
    for idx in best_indices:
        predicted_captions.append(train_descriptions[idx])
        
    # 3. Compute Metrics
    print("Computing BLEU-4...")
    bleu_scores = compute_bleu4(predicted_captions, val_descriptions_gt)
    
    print("Computing BERTScore...")
    bert_scores = compute_bertscore(predicted_captions, val_descriptions_gt)
    
    # Merge scores
    scores = {**bleu_scores, **bert_scores}
    
    return scores, predicted_captions, val_descriptions_gt

def main():
    # ... (rest of main function)
    print(f"Device: {DEVICE}")
    
    # Check files
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Train first using train_gcn.py")
        return
        
    if not os.path.exists(TRAIN_GRAPHS) or not os.path.exists(VAL_GRAPHS):
        print("Data files missing in 'data/' directory.")
        return

    # Load Train Embeddings (Database for retrieval)
    print(f"Loading train embeddings from {TRAIN_EMB_CSV}...")
    train_id2emb = load_id2emb(TRAIN_EMB_CSV)
    train_ids = list(train_id2emb.keys())
    
    # Convert to tensor
    train_embs_tensor = torch.stack([train_id2emb[id_] for id_ in train_ids]).to(DEVICE)
    train_embs_tensor = F.normalize(train_embs_tensor, dim=-1)
    emb_dim = train_embs_tensor.size(1)
    
    # Load Train Descriptions
    print(f"Loading train descriptions from {TRAIN_GRAPHS}...")
    train_id2desc = load_descriptions_from_graphs(TRAIN_GRAPHS)
    # Ensure order matches embeddings
    train_descriptions_ordered = [train_id2desc[id_] for id_ in train_ids]

    # Load Model
    print("Loading model...")
    model = MolGNN(out_dim=emb_dim).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    # Load Validation Data
    print(f"Loading validation graphs from {VAL_GRAPHS}...")
    # We pass dummy emb_dict just to make Dataset happy if needed, or None
    # PreprocessedGraphDataset constructor: (graph_path, id2emb=None)
    val_ds = PreprocessedGraphDataset(VAL_GRAPHS, emb_dict=None)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Run Evaluation
    scores, preds, refs = evaluate_bleu(
        model, 
        val_loader, 
        train_embs_tensor, 
        train_descriptions_ordered, 
        DEVICE
    )
    
    print("\n" + "="*40)
    print("BASELINE EVALUATION RESULTS (VALIDATION)")
    print("="*40)
    for metric, val in scores.items():
        print(f"{metric}: {val:.4f}")
    print("="*40)
    
    # Show examples
    print("\nSample Retrievals:")
    for i in range(5):
        print(f"\nGT:   {refs[i]}")
        print(f"Pred (Retrieved): {preds[i]}")

if __name__ == "__main__":
    main()
