/// <reference lib="webworker" />
// ECAPA ONNX 인퍼런스 워커 — 1초 웨이브폼(16000 float32) → 192차원 임베딩.
// 모델은 log-mel 특징추출·정규화·ECAPA가 모두 포함된 단일 그래프(public/models/ecapa.onnx).
import * as ort from 'onnxruntime-web'

const MODEL_URL = '/models/ecapa.onnx'
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
      const feeds = { wav: new ort.Tensor('float32', pcm, [1, 16000]) }
      const out = await sess.run(feeds)
      const emb = out.emb.data as Float32Array
      self.postMessage({ type: 'emb', emb: emb.slice(0) })
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
