#!/usr/bin/env python3
"""
train_gnn.py — Train a GAT-based SubgraphEncoder for loop closure embeddings.

Usage:
    python train_gnn.py --dataset dataset.pt --epochs 100 --lr 1e-3
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Batch


# ── Model ──

class SubgraphEncoder(nn.Module):
    """
    GAT-based encoder: subgraph → 128-dim L2-normalized embedding.

    Architecture:
        Linear(537, 128) → GATConv(128, 128) → GATConv(128, 128)
        → global_mean_pool → Linear(128, 128) → L2 normalize
    """

    def __init__(self, input_dim=537, hidden_dim=128, output_dim=128, heads=4):
        super().__init__()

        self.node_proj = nn.Linear(input_dim, hidden_dim)

        self.conv1 = GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                             dropout=0.1, add_self_loops=True)
        self.conv2 = GATConv(hidden_dim, hidden_dim // heads, heads=heads,
                             dropout=0.1, add_self_loops=True)

        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Project node features to hidden dim
        x = F.relu(self.node_proj(x))

        # GAT layers
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # Graph-level pooling
        x = global_mean_pool(x, batch)

        # Output projection + L2 normalize
        x = self.out_proj(x)
        x = F.normalize(x, p=2, dim=-1)

        return x


# ── Loss ──

def contrastive_loss(emb_a, emb_b, labels, margin=0.3):
    """
    Contrastive loss:
      - Positive pairs (label=1): minimize (1 - cosine_sim)^2
      - Negative pairs (label=0): minimize max(0, cosine_sim - margin)^2
    """
    cos_sim = F.cosine_similarity(emb_a, emb_b, dim=-1)

    pos_loss = labels * (1.0 - cos_sim) ** 2
    neg_loss = (1.0 - labels) * torch.clamp(cos_sim - margin, min=0) ** 2

    loss = (pos_loss + neg_loss).mean()
    return loss, cos_sim


# ── Training ──

def train_epoch(model, graphs, pairs, labels, optimizer, device, batch_size=64):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Shuffle pairs
    indices = np.random.permutation(len(pairs))

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_pairs = [pairs[i] for i in batch_idx]
        batch_labels = torch.tensor([labels[i] for i in batch_idx],
                                    dtype=torch.float32, device=device)

        # Create batched graphs for A and B
        graphs_a = [graphs[p[0]].to(device) for p in batch_pairs]
        graphs_b = [graphs[p[1]].to(device) for p in batch_pairs]

        batch_a = Batch.from_data_list(graphs_a)
        batch_b = Batch.from_data_list(graphs_b)

        optimizer.zero_grad()
        emb_a = model(batch_a)
        emb_b = model(batch_b)

        loss, _ = contrastive_loss(emb_a, emb_b, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, graphs, pairs, labels, device, batch_size=64):
    model.eval()
    all_sims = []
    all_labels = []

    for start in range(0, len(pairs), batch_size):
        batch_pairs = pairs[start:start + batch_size]
        batch_labels_np = labels[start:start + batch_size]

        graphs_a = [graphs[p[0]].to(device) for p in batch_pairs]
        graphs_b = [graphs[p[1]].to(device) for p in batch_pairs]

        batch_a = Batch.from_data_list(graphs_a)
        batch_b = Batch.from_data_list(graphs_b)

        emb_a = model(batch_a)
        emb_b = model(batch_b)

        sims = F.cosine_similarity(emb_a, emb_b, dim=-1)
        all_sims.extend(sims.cpu().numpy())
        all_labels.extend(batch_labels_np)

    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)

    # Compute metrics at threshold = 0.5
    preds = (all_sims > 0.5).astype(float)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    pos_sims = all_sims[all_labels == 1]
    neg_sims = all_sims[all_labels == 0]

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pos_sim_mean": pos_sims.mean() if len(pos_sims) > 0 else 0,
        "neg_sim_mean": neg_sims.mean() if len(neg_sims) > 0 else 0,
        "pos_sim_std": pos_sims.std() if len(pos_sims) > 0 else 0,
        "neg_sim_std": neg_sims.std() if len(neg_sims) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Train GNN SubgraphEncoder")
    parser.add_argument("--dataset", default="dataset.pt", help="Dataset file from build_dataset.py")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (pairs)")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--save_dir", default="models", help="Model save directory")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.dataset)
    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, weights_only=False)

    graphs = dataset["graphs"]
    pos_pairs = dataset["pos_pairs"]
    neg_pairs = dataset["neg_pairs"]

    print(f"  {len(graphs)} graphs, {len(pos_pairs)} pos pairs, {len(neg_pairs)} neg pairs")

    # Combine pairs and labels
    all_pairs = pos_pairs + neg_pairs
    all_labels = [1.0] * len(pos_pairs) + [0.0] * len(neg_pairs)

    # Train/val split
    np.random.seed(42)
    indices = np.random.permutation(len(all_pairs))
    val_size = int(len(indices) * args.val_split)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_pairs = [all_pairs[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    val_pairs = [all_pairs[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]

    print(f"  Train: {len(train_pairs)} pairs | Val: {len(val_pairs)} pairs")

    # Input dim from first graph
    input_dim = graphs[0].x.shape[1]
    print(f"  Node feature dim: {input_dim}")

    # Model
    model = SubgraphEncoder(input_dim=input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {param_count:,}")

    # Training loop
    best_f1 = 0.0
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'Epoch':>5} {'Train Loss':>12} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Pos Sim':>10} {'Neg Sim':>10}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, graphs, train_pairs, train_labels,
                                 optimizer, device, args.batch_size)
        scheduler.step()

        # Evaluate every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            metrics = evaluate(model, graphs, val_pairs, val_labels, device, args.batch_size)

            print(f"{epoch:5d} {train_loss:12.6f} "
                  f"{metrics['precision']:8.4f} {metrics['recall']:8.4f} {metrics['f1']:8.4f} "
                  f"{metrics['pos_sim_mean']:8.4f}±{metrics['pos_sim_std']:.2f} "
                  f"{metrics['neg_sim_mean']:8.4f}±{metrics['neg_sim_std']:.2f}")

            if metrics["f1"] >= best_f1:
                best_f1 = metrics["f1"]
                save_path = os.path.join(save_dir, "best_encoder.pt")
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "input_dim": input_dim,
                    "epoch": epoch,
                    "f1": best_f1,
                    "metrics": metrics,
                }, save_path)

    # Save final model
    final_path = os.path.join(save_dir, "final_encoder.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "epoch": args.epochs,
    }, final_path)

    print(f"\nTraining complete. Best F1: {best_f1:.4f}")
    print(f"  Best model: {os.path.join(save_dir, 'best_encoder.pt')}")
    print(f"  Final model: {final_path}")


if __name__ == "__main__":
    main()
