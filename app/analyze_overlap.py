"""
Script to analyze structural overlap between Test and Training sets.
Validates if test molecules are structurally identical or highly similar to training molecules.
"""

import argparse
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import numpy as np

from app.model import build_molecular_graph_model


def collate_graphs(batch):
    """Collate graphs into batch."""
    return Batch.from_data_list(batch)


def analyze_similarity(train_loader, test_loader, model, device):
    """
    Analyzes similarity between test and train sets.
    
    Args:
        train_loader: DataLoader for training graphs
        test_loader: DataLoader for test graphs
        model: Loaded Graph2CaptionModel
        device: Torch device
    """
    print("\n--- Starting Similarity Analysis ---")
    
    # 1. Extract Embeddings
    model.eval()
    encoder = model.encoder
    
    train_embs_list = []
    test_embs_list = []
    
    print("Extracting Training Embeddings...")
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Train Embs"):
            batch = batch.to(device)
            # Get graph-level embeddings [B, hidden_dim]
            emb = encoder(batch, return_node_embeddings=False)
            train_embs_list.append(emb.cpu())

    print("Extracting Test Embeddings...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test Embs"):
            batch = batch.to(device)
            emb = encoder(batch, return_node_embeddings=False)
            test_embs_list.append(emb.cpu())
            
    # Concatenate
    train_embs = torch.cat(train_embs_list, dim=0)
    test_embs = torch.cat(test_embs_list, dim=0)
    
    print(f"\nEmbeddings Shape: Train={train_embs.shape}, Test={test_embs.shape}")
    
    # 2. L2 Normalization
    print("Applying L2 Normalization...")
    train_embs = F.normalize(train_embs, p=2, dim=1)
    test_embs = F.normalize(test_embs, p=2, dim=1)
    
    # 3. Compute Similarity Matrix (Test x Train^T)
    # Using batches to avoid OOM if matrices are huge, but for this size full matmul might fit
    # If OOM happens, we can loop over test rows.
    
    print("Computing Similarity Matrix...")
    # Move validation/test to GPU for fast matmul if possible, or do it on CPU
    if torch.cuda.is_available() and train_embs.shape[0] < 100000: # Heuristic limit
        train_embs = train_embs.to(device)
        test_embs = test_embs.to(device)
        
    similarity_matrix = torch.matmul(test_embs, train_embs.t())
    
    # 4. Nearest Neighbor Search (Max Similarity)
    print("Finding Nearest Neighbors...")
    max_sims, _ = similarity_matrix.max(dim=1)
    
    # Move back to CPU/Numpy for stats
    max_sims = max_sims.cpu().numpy()
    
    # 5. Statistical Reporting
    avg_max_sim = np.mean(max_sims)
    count_90 = np.sum(max_sims > 0.90)
    count_95 = np.sum(max_sims > 0.95)
    count_99 = np.sum(max_sims > 0.99)
    total_test = len(max_sims)
    
    print("\n" + "="*40)
    print("ANALYSIS RESULTS")
    print("="*40)
    print(f"Average Max Similarity: {avg_max_sim:.4f}")
    print(f"Matches > 0.90: {count_90} ({100*count_90/total_test:.2f}%)")
    print(f"Matches > 0.95: {count_95} ({100*count_95/total_test:.2f}%)")
    print(f"Matches > 0.99: {count_99} ({100*count_99/total_test:.2f}%)")
    print("="*40)
    
    # 6. Visualization
    plt.figure(figsize=(10, 6))
    plt.hist(max_sims, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(x=0.90, color='red', linestyle='--', linewidth=2, label='Threshold (0.90)')
    plt.axvline(x=avg_max_sim, color='green', linestyle='-', linewidth=2, label=f'Mean ({avg_max_sim:.2f})')
    
    plt.title('Distribution of Max Similarity (Test vs Train)', fontsize=14)
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Count of Test Molecules', fontsize=12)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    output_png = "similarity_histogram.png"
    plt.savefig(output_png)
    print(f"\nHistogram saved to {output_png}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Training Data
    print(f"Loading training graphs from {args.data_dir}/train_graphs.pkl...")
    with open(f"{args.data_dir}/train_graphs.pkl", "rb") as f:
        train_graphs = pickle.load(f)
        
    # Load Test Data
    print(f"Loading test graphs from {args.data_dir}/test_graphs.pkl...")
    with open(f"{args.data_dir}/test_graphs.pkl", "rb") as f:
        test_graphs = pickle.load(f)
        
    if args.subset:
        train_graphs = train_graphs[:args.subset]
        test_graphs = test_graphs[:args.subset]
        
    # DataLoaders
    train_loader = DataLoader(
        train_graphs, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_graphs
    )
    
    test_loader = DataLoader(
        test_graphs, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_graphs
    )
    
    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = build_molecular_graph_model(
        lm_name=args.lm_name,
        freeze_lm=False,
        gradient_checkpointing=False,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    # Run Analysis
    analyze_similarity(train_loader, test_loader, model, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Test/Train Overlap")
    
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to data directory")
    parser.add_argument("--lm_name", type=str, default="laituan245/molt5-base")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--subset", type=int, default=None, help="Use subset for debugging")
    
    # LoRA args (needed for model loading)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    
    args = parser.parse_args()
    main(args)