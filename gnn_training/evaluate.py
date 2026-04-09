#!/usr/bin/env python3
"""
evaluate.py — Evaluate trained GNN encoder: similarity distributions + metrics.

Usage:
    python evaluate.py --dataset dataset.pt --model models/best_encoder.pt
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# Import model from train_gnn
from train_gnn import SubgraphEncoder


@torch.no_grad()
def compute_all_embeddings(model, graphs, device, batch_size=32):
    """Compute embeddings for all graphs."""
    model.eval()
    all_embeddings = []

    for start in range(0, len(graphs), batch_size):
        batch_graphs = [graphs[i].to(device) for i in range(start, min(start + batch_size, len(graphs)))]
        batch = Batch.from_data_list(batch_graphs)
        emb = model(batch)
        all_embeddings.append(emb.cpu())

    return torch.cat(all_embeddings, dim=0)


def find_best_threshold(pos_sims, neg_sims):
    """Find threshold that maximizes F1."""
    best_f1 = 0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 0.95, 0.05):
        tp = (pos_sims >= thresh).sum()
        fp = (neg_sims >= thresh).sum()
        fn = (pos_sims < thresh).sum()

        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-8)

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN encoder")
    parser.add_argument("--dataset", default="dataset.pt")
    parser.add_argument("--model", default="models/best_encoder.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = torch.load(os.path.join(base_dir, args.dataset), weights_only=False)
    graphs = dataset["graphs"]
    pos_pairs = dataset["pos_pairs"]
    neg_pairs = dataset["neg_pairs"]
    gt_positions = dataset["gt_positions"]

    # Load model
    checkpoint = torch.load(os.path.join(base_dir, args.model), weights_only=False)
    model = SubgraphEncoder(input_dim=checkpoint["input_dim"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from epoch {checkpoint.get('epoch', '?')}")

    # Compute all embeddings
    print("Computing embeddings...")
    embeddings = compute_all_embeddings(model, graphs, device)
    print(f"  Embeddings shape: {embeddings.shape}")

    # Compute similarities for positive and negative pairs
    pos_sims = np.array([
        F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        for i, j in pos_pairs
    ])
    neg_sims = np.array([
        F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        for i, j in neg_pairs
    ])

    # Print distribution stats
    print(f"\n{'='*60}")
    print(f"SIMILARITY DISTRIBUTIONS")
    print(f"{'='*60}")
    print(f"  Positive pairs ({len(pos_sims)}):")
    if len(pos_sims) > 0:
        print(f"    Mean: {pos_sims.mean():.4f} ± {pos_sims.std():.4f}")
        print(f"    Min:  {pos_sims.min():.4f}  Max: {pos_sims.max():.4f}")
    else:
        print(f"    (none)")
    print(f"  Negative pairs ({len(neg_sims)}):")
    if len(neg_sims) > 0:
        print(f"    Mean: {neg_sims.mean():.4f} ± {neg_sims.std():.4f}")
        print(f"    Min:  {neg_sims.min():.4f}  Max: {neg_sims.max():.4f}")
    else:
        print(f"    (none)")
    if len(pos_sims) > 0 and len(neg_sims) > 0:
        print(f"  Separation: {pos_sims.mean() - neg_sims.mean():.4f}")
    else:
        print(f"  Separation: N/A (need both positive and negative pairs)")

    # Find best threshold
    best_thresh, best_f1 = find_best_threshold(pos_sims, neg_sims)

    # Metrics at best threshold
    tp = (pos_sims >= best_thresh).sum()
    fp = (neg_sims >= best_thresh).sum()
    fn = (pos_sims < best_thresh).sum()
    tn = (neg_sims < best_thresh).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    print(f"\n{'='*60}")
    print(f"METRICS AT BEST THRESHOLD = {best_thresh:.2f}")
    print(f"{'='*60}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Metrics at various thresholds
    print(f"\n{'='*60}")
    print(f"METRICS AT VARIOUS THRESHOLDS")
    print(f"{'='*60}")
    print(f"  {'Thresh':>7}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        tp_t = (pos_sims >= thresh).sum()
        fp_t = (neg_sims >= thresh).sum()
        fn_t = (pos_sims < thresh).sum()
        p = tp_t / max(tp_t + fp_t, 1)
        r = tp_t / max(tp_t + fn_t, 1)
        f = 2 * p * r / max(p + r, 1e-8)
        print(f"  {thresh:7.2f}  {p:7.4f}  {r:7.4f}  {f:7.4f}  {tp_t:5d}  {fp_t:5d}")

    # Save distributions for plotting
    results_path = os.path.join(base_dir, "eval_results.npz")
    np.savez(results_path,
             pos_sims=pos_sims, neg_sims=neg_sims,
             embeddings=embeddings.numpy(),
             gt_positions=gt_positions,
             best_threshold=best_thresh)
    print(f"\nResults saved to {results_path}")

    # Simple text histogram
    print(f"\n{'='*60}")
    print(f"SIMILARITY HISTOGRAM")
    print(f"{'='*60}")
    bins = np.arange(-0.2, 1.05, 0.1)
    pos_hist, _ = np.histogram(pos_sims, bins=bins)
    neg_hist, _ = np.histogram(neg_sims, bins=bins)

    max_count = max(pos_hist.max(), neg_hist.max(), 1)
    for i in range(len(bins) - 1):
        bar_pos = '█' * int(pos_hist[i] / max_count * 30)
        bar_neg = '░' * int(neg_hist[i] / max_count * 30)
        print(f"  [{bins[i]:+.1f},{bins[i+1]:+.1f})  POS: {bar_pos:<30} ({pos_hist[i]:3d})")
        print(f"  {'':>15}  NEG: {bar_neg:<30} ({neg_hist[i]:3d})")


if __name__ == "__main__":
    main()
