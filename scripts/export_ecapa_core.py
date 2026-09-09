"""ECAPA 코어(정규화+ECAPA) ONNX 추출. 특징(log-mel)은 브라우저 JS에서 계산.

onnxruntime-web은 STFT 커널이 없어서 log-mel(STFT+mel+dB)을 JS로 옮기고,
이 ONNX는 입력 (1,101,80) mel 프레임 → (1,1,192) 임베딩만 담는다.
JS 특징 계산은 scripts/verify_mel.py로 이 파이프라인과 1e-3 수준으로 일치함을 확인했다.

    .venv/bin/python ../../bstudio-web/scripts/export_ecapa_core.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/data/workspace/ai2breath/ai2breath-train")
from ai2breath_train.encoder import SpeakerEncoder

OUT = Path("/data/workspace/bstudio-web/public/models")
ONNX = OUT / "ecapa_core.onnx"


class CorePipe(torch.nn.Module):
    def __init__(self, mods) -> None:
        super().__init__()
        self.norm = mods.mean_var_norm
        self.emb = mods.embedding_model

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        lens = torch.ones(feats.shape[0], device=feats.device)
        n = self.norm(feats, lens)
        return self.emb(n, lens)


def main() -> int:
    enc = SpeakerEncoder("ecapa", device="cpu")
    pipe = CorePipe(enc.model.mods).eval()
    dummy = torch.zeros(1, 101, 80)
    torch.onnx.export(
        pipe,
        dummy,
        str(ONNX),
        input_names=["feats"],
        output_names=["emb"],
        dynamic_axes=None,
        opset_version=17,
    )
    # 재인라인(외부 데이터 방지)
    import onnx

    m = onnx.load(str(ONNX), load_external_data=True)
    onnx.save(m, str(ONNX), save_as_external_data=False)
    print(f"onnx core: {ONNX} ({ONNX.stat().st_size/1e6:.1f} MB)")

    # parity: 전체 파이프라인과 비교
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    x = np.random.RandomState(0).randn(16000).astype(np.float32)
    with torch.inference_mode():
        feats = enc.model.mods.compute_features(torch.from_numpy(x[None, :])).numpy()
        ref = enc.model.encode_batch(torch.from_numpy(x[None, :])).numpy()
    y = sess.run(None, {"feats": feats})[0]
    print("core onnx max diff (given torch feats):", float(np.abs(y - ref).max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
