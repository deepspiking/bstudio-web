# bstudio-web — 언어 투영 (실시간, 브라우저 on-device 추론)

ecapa 화자 임베딩을 LDA로 x축(+/−)에 나눈 실시간 2D 화면의 독립 버전.
위 2D(배경 vocal-tract.svg + 라이브 펄스), 아래는 녹음될 때마다
쌓이는 "파형 + 언어신호(위 +/아래 −)" 줄. 파형 줄을 클릭하면 그 위치부터 재생.

- **추론은 전부 브라우저에서**: ECAPA(log-mel+norm 포함)를 ONNX로 export해
  Web Worker(onnxruntime-web)가 1초 창마다 임베딩 → 로컬에서 선형 투영.
  별도 live 서버가 필요 없다. (모델 export: `scripts/export_ecapa.py`)
- 모델·wasm은 저장소에 포함하지 않는다 — `public/models/`에 직접 두어야 한다:
  `ecapa_core.onnx`(약 84MB)와 `ort-wasm-simd-threaded.wasm`.
- 배경 그림: `public/vocal-tract.svg`
- 마이크는 secure context가 필요 → `HTTPS=1 npm run dev` (인증서 경로는
  `CERT_DIR` 환경변수로 지정, 기본은 이 머신의 공유 인증서).

```bash
npm install
CERT_DIR=/path/to/certs HTTPS=1 npm run dev -- --port 8442 --host 0.0.0.0
```

## 발화(EPD)

- 게이트: `EPD_DBFS = -50` (이상만 발성으로 취급) — `src/components/OralTractLive.tsx`
- 0.4s 이상 발성이 쌓인 뒤 0.5s 무음이 이어지면 발화 종료(자동 캡처).
- 프레임 간 0.18s 이상 벌어지면 VAD 끊김 — 선을 잇지 않는다.
- 재생 표시는 보간 없이 마지막 발화 프레임 값을 유지.

