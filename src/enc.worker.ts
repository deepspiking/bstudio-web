/// <reference lib="webworker" />
// 브라우저 on-device 추론: JS log-mel(mel.ts) → ECAPA 코어 ONNX → 192차원 임베딩.
import { audioToMel } from './mel'

const MODEL_URL = '/models/ecapa_core.onnx'

let session: Awaited<ReturnType<typeof makeSession>> | null = null

async function makeSession() {
  const ort = await import('onnxruntime-web')
  ort.env.wasm.wasmPaths = `${self.location.origin}/models/`
  ort.env.wasm.numThreads = 1
  const sess = await ort.InferenceSession.create(MODEL_URL, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all',
  })
  return { ort, sess }
}

async function getSession() {
  if (!session) session = await makeSession()
  return session
}

self.onmessage = async (e: MessageEvent) => {
  try {
    const data = e.data
    if (data && data.type === 'load') {
      await getSession()
      self.postMessage({ type: 'ready' })
      return
    }
    if (data && data.type === 'infer' && data.pcm) {
      const { ort, sess } = await getSession()
      const feats = audioToMel(data.pcm as Float32Array)
      const feeds = { feats: new ort.Tensor('float32', feats, [1, 101, 80]) }
      const out = await sess.run(feeds)
      const emb = out.emb.data as Float32Array
      self.postMessage({ type: 'emb', emb: emb.slice(0) })
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: err instanceof Error ? err.message : String(err) })
  }
}
