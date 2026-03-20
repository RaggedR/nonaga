"""Export PyTorch model to ONNX format for browser deployment."""

import torch
import numpy as np
import argparse
import os


def export_model(checkpoint_path, output_path="web/model.onnx"):
    """Export a trained NonagaNet to ONNX."""
    from model.network import NonagaNet, GRID_SIZE, INPUT_CHANNELS

    net = NonagaNet()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()

    # Dummy input
    dummy = torch.randn(1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)

    # Export
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.onnx.export(
        net,
        dummy,
        output_path,
        input_names=["board"],
        output_names=["piece_policy", "tile_policy", "value"],
        dynamic_axes={
            "board": {0: "batch"},
            "piece_policy": {0: "batch"},
            "tile_policy": {0: "batch"},
            "value": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"Exported to {output_path}")

    # Verify
    verify_onnx(net, output_path)


def verify_onnx(net, onnx_path):
    """Verify ONNX model produces same output as PyTorch."""
    import onnxruntime as ort
    from model.network import GRID_SIZE, INPUT_CHANNELS

    test_input = np.random.randn(1, INPUT_CHANNELS, GRID_SIZE, GRID_SIZE).astype(np.float32)

    # PyTorch output
    net.eval()
    with torch.no_grad():
        pt_pp, pt_tp, pt_v = net(torch.from_numpy(test_input))
        pt_pp = pt_pp.numpy()
        pt_tp = pt_tp.numpy()
        pt_v = pt_v.numpy()

    # ONNX output
    sess = ort.InferenceSession(onnx_path)
    onnx_pp, onnx_tp, onnx_v = sess.run(None, {"board": test_input})

    # Compare
    assert np.allclose(pt_pp, onnx_pp, atol=1e-5), "Piece policy mismatch"
    assert np.allclose(pt_tp, onnx_tp, atol=1e-5), "Tile policy mismatch"
    assert np.allclose(pt_v, onnx_v, atol=1e-5), "Value mismatch"
    print("ONNX verification passed: outputs match PyTorch")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output", default="web/model.onnx", help="Output ONNX path")
    args = parser.parse_args()
    export_model(args.checkpoint, args.output)
