/// <reference lib="webworker" />
// 브라우저 on-device 추론: JS log-mel → ECAPA 코어(정규화+ECAPA) ONNX → 192차원 임베딩.
// log-mel은 onnxruntime-web에 없는 STFT를 피하기 위해 JS로 직접 계산한다(mel.ts).
import * as ort from 'onnxruntime-web'
import { audioToMel } from './mel'

const MODEL_URL = '/models/ecapa_core.onnx'
let session: ort.InferenceSession | null = null

ort.env.wasm.wasmPaths = `${self.location.origin}/models/`
ort.env.wasm.numThreads = 1

async function load() {
  if (!session) {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })
  }
  return session
}

self.onmessage = async (e: MessageEvent) => {
  try {
    const data = e.data
    if (data && data.type === 'load') {
      await load()
      self.postMessage({ type: 'ready' })
      return
    }
    if (data && data.type === 'infer' && data.pcm) {
      const sess = await load()
      const pcm = data.pcm as Float32Array
      const feats = audioToMel(pcm)
      const feeds = { feats: new ort.Tensor('float32', feats, [1, 101, 80]) }
      const out = await sess.run(feeds)
      const emb = out.emb.data as Float32Array
      self.postMessage({ type: 'emb', emb: emb.slice(0) })
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
