#!/usr/bin/env python3
"""
export_weights.py — Export GNN weights as raw binary files for C++ Eigen loading.

Usage:
    python export_weights.py --model models/best_encoder.pt --output_dir /models/gnn_weights
"""

import argparse
import os
import struct
import numpy as np
import torch


def save_tensor(path, tensor):
    """Save tensor as raw float32 binary with shape header."""
    t = tensor.detach().cpu().float().numpy()
    with open(path, 'wb') as f:
        # Write ndim, then each dim, then raw data
        f.write(struct.pack('i', t.ndim))
        for d in t.shape:
            f.write(struct.pack('i', d))
        f.write(t.tobytes())
    print(f"  {path}: shape={t.shape}, {os.path.getsize(path)} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best_encoder.pt")
    parser.add_argument("--output_dir", default="/models/gnn_weights")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(os.path.join(base_dir, args.model), weights_only=False)
    sd = checkpoint['model_state_dict']
    input_dim = checkpoint['input_dim']

    print(f"Input dim: {input_dim}")
    print(f"Epoch: {checkpoint.get('epoch', '?')}")
    print(f"\nExporting weights to {args.output_dir}/")

    # node_proj: Linear(517, 128)
    save_tensor(os.path.join(args.output_dir, "node_proj_weight.bin"), sd['node_proj.weight'])
    save_tensor(os.path.join(args.output_dir, "node_proj_bias.bin"), sd['node_proj.bias'])

    # conv1: GATConv(128, 32, heads=4)
    # PyG stores: lin_src.weight [heads*out, in] = [128, 128]
    #             att_src [1, heads, out] = [1, 4, 32]
    #             att_dst [1, heads, out] = [1, 4, 32]
    for layer in ['conv1', 'conv2']:
        if f'{layer}.lin_src.weight' in sd:
            w = sd[f'{layer}.lin_src.weight']
        elif f'{layer}.lin.weight' in sd:
            w = sd[f'{layer}.lin.weight']
        else:
            raise KeyError(f"Cannot find weight for {layer}")

        save_tensor(os.path.join(args.output_dir, f"{layer}_W.bin"), w)
        save_tensor(os.path.join(args.output_dir, f"{layer}_att_src.bin"),
                    sd[f'{layer}.att_src'].squeeze(0))  # [heads, out_dim]
        save_tensor(os.path.join(args.output_dir, f"{layer}_att_dst.bin"),
                    sd[f'{layer}.att_dst'].squeeze(0))  # [heads, out_dim]

    # out_proj: Linear(128, 128)
    save_tensor(os.path.join(args.output_dir, "out_proj_weight.bin"), sd['out_proj.weight'])
    save_tensor(os.path.join(args.output_dir, "out_proj_bias.bin"), sd['out_proj.bias'])

    # Save model config
    config_path = os.path.join(args.output_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"input_dim {input_dim}\n")
        f.write(f"hidden_dim 128\n")
        f.write(f"output_dim 128\n")
        f.write(f"heads 4\n")
    print(f"  {config_path}: model config")

    print(f"\n✓ All weights exported ({len(os.listdir(args.output_dir))} files)")


if __name__ == "__main__":
    main()
