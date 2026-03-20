"""
Export trained PyTorch model weights as binary + manifest for browser deployment.

Output:
  weights.bin  — concatenated float32 arrays (~2.3 MB, ~1.5 MB gzipped)
  manifest.json — tensor name → {shape, offset, length}

The browser loads both, slices the ArrayBuffer into typed arrays,
and implements the forward pass in pure JS. No ONNX runtime needed.
"""

import json
import struct
import torch
import numpy as np
import argparse
import os


def export_weights(checkpoint_path, output_dir="web"):
    """Export model weights as binary + manifest."""
    from model.network import NonagaNet

    net = NonagaNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    os.makedirs(output_dir, exist_ok=True)
    bin_path = os.path.join(output_dir, "weights.bin")
    manifest_path = os.path.join(output_dir, "manifest.json")

    manifest = {}
    offset = 0

    with open(bin_path, 'wb') as f:
        for name, param in net.state_dict().items():
            arr = param.detach().cpu().numpy().astype(np.float32).flatten()
            f.write(arr.tobytes())
            manifest[name] = {
                'shape': list(param.shape),
                'offset': offset,
                'length': arr.size,
            }
            offset += arr.size

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    bin_size = os.path.getsize(bin_path) / (1024 * 1024)
    manifest_size = os.path.getsize(manifest_path) / 1024
    print(f"Exported to {output_dir}/")
    print(f"  weights.bin:   {bin_size:.1f} MB ({offset:,} floats)")
    print(f"  manifest.json: {manifest_size:.1f} KB ({len(manifest)} tensors)")

    # Verify
    verify_weights(net, bin_path, manifest_path)


def verify_weights(net, bin_path, manifest_path):
    """Verify exported weights match the model."""
    with open(manifest_path) as f:
        manifest = json.load(f)
    with open(bin_path, 'rb') as f:
        raw = f.read()
    all_floats = np.frombuffer(raw, dtype=np.float32)

    state_dict = net.state_dict()
    for name, param in state_dict.items():
        info = manifest[name]
        loaded = all_floats[info['offset']:info['offset'] + info['length']]
        loaded = loaded.reshape(info['shape'])
        orig = param.detach().cpu().numpy()
        assert np.allclose(orig, loaded, atol=1e-6), f"Mismatch: {name}"

    print(f"Verification passed: all {len(manifest)} tensors match")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output-dir", default="web", help="Output directory")
    args = parser.parse_args()
    export_weights(args.checkpoint, args.output_dir)
