"""ECAPA(spkrec-ecapa-voxceleb)를 웨이브폼→임베딩 단일 ONNX로 추출한다.

입력: (1, 16000) float32 1초 웨이브폼 → 출력: (1, 1, 192) 임베딩.
특징(log-mel 80) + 배치 내 정규화 + ECAPA까지 한 그래프에 포함하므로
브라우저(onnxruntime-web)에서 원음만 넣으면 된다.

    .venv/bin/python ../../bstudio-web/scripts/export_ecapa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/workspace/ai2breath/ai2breath-train")
from ai2breath_train.encoder import SpeakerEncoder

OUT = Path("/data/workspace/bstudio-web/public/models")
OUT.mkdir(parents=True, exist_ok=True)
ONNX = OUT / "ecapa.onnx"
INT8 = OUT / "ecapa_int8.onnx"


class EcapaPipe(torch.nn.Module):
    def __init__(self, mods) -> None:
        super().__init__()
        self.feats = mods.compute_features
        self.norm = mods.mean_var_norm
        self.emb = mods.embedding_model

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        lens = torch.ones(wav.shape[0], device=wav.device)
        f = self.feats(wav)
        n = self.norm(f, lens)
        return self.emb(n, lens)


def main() -> int:
    enc = SpeakerEncoder("ecapa", device="cpu")
    pipe = EcapaPipe(enc.model.mods).eval()

    dummy = torch.zeros(1, 16000)
    torch.onnx.export(
        pipe,
        dummy,
        str(ONNX),
        input_names=["wav"],
        output_names=["emb"],
        dynamic_axes=None,
        opset_version=17,
    )
    print(f"onnx: {ONNX} ({ONNX.stat().st_size/1e6:.1f} MB)")

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(ONNX), so, providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(1, 16000).astype(np.float32)
    y = sess.run(None, {"wav": x})[0]
    with torch.inference_mode():
        ref = enc.model.encode_batch(torch.from_numpy(x)).numpy()
    print("onnx max diff:", float(np.abs(y - ref).max()))

    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(ONNX), str(INT8), weight_type=QuantType.QInt8)
    print(f"int8: {INT8} ({INT8.stat().st_size/1e6:.1f} MB)")
    sess8 = ort.InferenceSession(str(INT8), so, providers=["CPUExecutionProvider"])
    y8 = sess8.run(None, {"wav": x})[0]
    print("int8 max diff:", float(np.abs(y8 - ref).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
