#!/usr/bin/env python3
"""
export_model.py — Export trained GNN encoder to TorchScript for C++ inference.

Since PyG's GATConv uses dynamic scatter ops that are hard to trace directly,
we re-implement the forward pass using pure PyTorch ops that TorchScript can handle.

Usage (run OUTSIDE container, in conda env with PyG):
    python export_model.py --model models/best_encoder.pt --output /models/gnn_encoder_scripted.pt
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class PureGATLayer(nn.Module):
    """
    Pure PyTorch implementation of a single GAT layer (multi-head attention).
    No PyG dependencies — fully TorchScript-compatible.
    """

    def __init__(self, in_dim, out_dim, heads=4, negative_slope=0.2):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim  # per-head output dim
        self.negative_slope = negative_slope

        # Linear projection: in_dim → heads * out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)

        # Attention parameters: a^T [Wh_i || Wh_j] → 2 * out_dim per head
        self.att_src = nn.Parameter(torch.zeros(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.zeros(1, heads, out_dim))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim]
            edge_index: [2, E]
        Returns:
            [N, heads * out_dim]
        """
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]

        # Project: [N, in_dim] → [N, heads, out_dim]
        h = self.W(x).view(N, self.heads, self.out_dim)

        # Attention scores
        # src attention: a_src^T * Wh_i for source nodes
        # dst attention: a_dst^T * Wh_j for destination nodes
        e_src = (h * self.att_src).sum(dim=-1)  # [N, heads]
        e_dst = (h * self.att_dst).sum(dim=-1)  # [N, heads]

        # For each edge (i→j): score = LeakyReLU(e_src[i] + e_dst[j])
        edge_score = F.leaky_relu(
            e_src[src] + e_dst[dst],  # [E, heads]
            negative_slope=self.negative_slope
        )

        # Softmax over incoming edges for each node
        # Manual sparse softmax: exp(score) / sum_neighbors(exp(score))
        edge_score_exp = torch.exp(edge_score - edge_score.max())  # numerical stability

        # Sum of exp scores per destination node: [N, heads]
        denom = torch.zeros(N, self.heads, device=x.device, dtype=x.dtype)
        denom.scatter_add_(0, dst.unsqueeze(1).expand(-1, self.heads), edge_score_exp)

        # Normalize: alpha[e] = exp(score[e]) / denom[dst[e]]
        alpha = edge_score_exp / (denom[dst] + 1e-8)  # [E, heads]

        # Weighted aggregation: for each node j, sum alpha[e] * h[src[e]]
        h_src = h[src]  # [E, heads, out_dim]
        weighted = h_src * alpha.unsqueeze(-1)  # [E, heads, out_dim]

        out = torch.zeros(N, self.heads, self.out_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(0,
                         dst.unsqueeze(1).unsqueeze(2).expand(-1, self.heads, self.out_dim),
                         weighted)

        # Concat heads: [N, heads * out_dim]
        return out.view(N, self.heads * self.out_dim)


class GNNEncoderScriptable(nn.Module):
    """
    TorchScript-compatible GNN encoder.
    Same architecture as SubgraphEncoder but using pure PyTorch GAT layers.
    """

    def __init__(self, input_dim: int = 537, hidden_dim: int = 128,
                 output_dim: int = 128, heads: int = 4):
        super().__init__()
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = PureGATLayer(hidden_dim, hidden_dim // heads, heads=heads)
        self.conv2 = PureGATLayer(hidden_dim, hidden_dim // heads, heads=heads)
        self.out_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] node features
            edge_index: [2, E] edge indices
            batch: [N] batch assignment (0 for single graph)
        Returns:
            [B, output_dim] L2-normalized graph embeddings
        """
        # Project node features
        x = F.relu(self.node_proj(x))

        # Add self-loops to edge_index
        N = x.size(0)
        self_loops = torch.arange(N, device=x.device).unsqueeze(0).expand(2, -1)
        edge_index_with_loops = torch.cat([edge_index, self_loops], dim=1)

        # GAT layers
        x = F.relu(self.conv1(x, edge_index_with_loops))
        x = F.relu(self.conv2(x, edge_index_with_loops))

        # Global mean pool per graph
        num_graphs = int(batch.max().item()) + 1
        graph_embs = torch.zeros(num_graphs, x.size(1), device=x.device, dtype=x.dtype)
        counts = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)

        graph_embs.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
        counts.scatter_add_(0, batch.unsqueeze(1), torch.ones(N, 1, device=x.device))
        graph_embs = graph_embs / (counts + 1e-8)

        # Output projection + L2 normalize
        graph_embs = self.out_proj(graph_embs)
        graph_embs = F.normalize(graph_embs, p=2.0, dim=-1)

        return graph_embs


def transfer_weights(pyg_model_state, scriptable_model):
    """Transfer weights from PyG SubgraphEncoder to scriptable model."""
    sd = scriptable_model.state_dict()

    # node_proj and out_proj are identical
    sd['node_proj.weight'] = pyg_model_state['node_proj.weight']
    sd['node_proj.bias'] = pyg_model_state['node_proj.bias']
    sd['out_proj.weight'] = pyg_model_state['out_proj.weight']
    sd['out_proj.bias'] = pyg_model_state['out_proj.bias']

    # GATConv weights:
    # PyG GATConv stores: lin_src.weight [heads*out, in], att_src [1, heads, out], att_dst
    # Our PureGATLayer stores: W.weight [heads*out, in], att_src [1, heads, out], att_dst
    for layer_idx, layer_name in enumerate(['conv1', 'conv2']):
        pyg_prefix = f'{layer_name}'

        # Linear weight: PyG has lin_src.weight
        if f'{pyg_prefix}.lin_src.weight' in pyg_model_state:
            sd[f'{layer_name}.W.weight'] = pyg_model_state[f'{pyg_prefix}.lin_src.weight']
        elif f'{pyg_prefix}.lin.weight' in pyg_model_state:
            sd[f'{layer_name}.W.weight'] = pyg_model_state[f'{pyg_prefix}.lin.weight']

        # Attention parameters
        sd[f'{layer_name}.att_src'] = pyg_model_state[f'{pyg_prefix}.att_src']
        sd[f'{layer_name}.att_dst'] = pyg_model_state[f'{pyg_prefix}.att_dst']

    scriptable_model.load_state_dict(sd)
    return scriptable_model


def main():
    parser = argparse.ArgumentParser(description="Export GNN encoder to TorchScript")
    parser.add_argument("--model", default="models/best_encoder.pt",
                        help="Trained model checkpoint")
    parser.add_argument("--output", default="/models/gnn_encoder_scripted.pt",
                        help="Output TorchScript model path")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Load checkpoint
    checkpoint = torch.load(os.path.join(base_dir, args.model), weights_only=False)
    input_dim = checkpoint['input_dim']
    print(f"Input dim: {input_dim}")
    print(f"Training epoch: {checkpoint.get('epoch', '?')}")

    # Create scriptable model and transfer weights
    scriptable = GNNEncoderScriptable(input_dim=input_dim)
    scriptable = transfer_weights(checkpoint['model_state_dict'], scriptable)
    scriptable.eval()

    # Verify with a dummy input
    N, E = 8, 20
    dummy_x = torch.randn(N, input_dim)
    dummy_edge_index = torch.randint(0, N, (2, E))
    dummy_batch = torch.zeros(N, dtype=torch.long)

    with torch.no_grad():
        out = scriptable(dummy_x, dummy_edge_index, dummy_batch)
        print(f"Output shape: {out.shape}")  # [1, 128]
        print(f"Output norm: {out.norm().item():.4f}")  # should be ~1.0

    # Script the model
    scripted = torch.jit.script(scriptable)

    # Verify scripted model produces same output
    with torch.no_grad():
        out_scripted = scripted(dummy_x, dummy_edge_index, dummy_batch)
        diff = (out - out_scripted).abs().max().item()
        print(f"Max diff (original vs scripted): {diff:.8f}")

    # Save
    scripted.save(args.output)
    print(f"\n✓ TorchScript model saved to: {args.output}")
    print(f"  Size: {os.path.getsize(args.output) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
