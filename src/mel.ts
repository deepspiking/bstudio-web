// speechbrain Fbank(ecapa)와 같은 log-mel 계산 — constants는 모델에서 추출
// DFT(400pt, reflect 아님 constant pad) → power → mel 삼각 필터 → 10log10 → top_db 80
const fromB64 = (b: string) => {
  const raw = atob(b)
  const arr = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) arr[i] = raw.charCodeAt(i)
  return new Float32Array(arr.buffer)
}
export const WIN = fromB64("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")
export const F_CENTRAL = fromB64("7vWwQafBM0Jl9YhC0Ia5QiCh60JcqA9DQVEqQ8fRRUPQMGJDWHV/Q1HTjkMWZp5D2HauQ5EJv0NaItBDdcXhQzv380MYXgNEfQwNRDUJF0S3ViFEkfcrRGfuNkTsPUJE9uhNRGHyWUQsXWZEaCxzRKMxgESFAodEiwqORHJLlUQGx5xEI3+kRLJ1rEStrLREGia9RBbkxUTJ6M5EdTbYRGDP4UTutetElez1RO06AEUwqgVFZ0ULRfoNEUVVBRdF9ywdRWCGI0UmEypF6NQwRVTNN0Ul/j5FIGlGRR0QTkUB9VVFwBleRWSAZkX7Km9FrRt4RVmqgEUrbIVFd1SKRXZkj0VqnZRFnACaRWaPn0UjS6VFQjWrRTpPsUWPmrdFzhi+RZTLxEWLtMtFZNXSRewv2kXtxeFFT5npRQKs8UU=")
export const BAND = fromB64("7vWwQWCNtkFGUrxBrEXCQUBpyEFgvs5BKEfVQTAE3EFI+OJBQCTqQVCK8UFQLPlBEIYAQsiVBEJIxghC2BgNQjCOEUKoJxZCUOYaQoDLH0Ig2CRCoA0qQmBtL0JQ+DRCoLA6QrCWQEKwrEZCwPNMQuBtU0JAHFpCwABhQuAcaEKAcm9CoAN3QuDRfkKwb4NC0JaHQsDfi0IwS5BCwNqUQrCOmULgaJ5CcGqjQlCUqEJg6K1C4GazQmASuUJg675CQPTEQiAty0LAmNFCQDjYQoAN30IgGuZCYF/tQqDf9EKAnPxC8EsCQ0BqBkNwqQpDIAsPQ1CQE0NAOhhDgAkdQ+D/IUOAHidDQGYsQ0DZMUOgdzdD4EM9QwA/Q0OgaklD4MdPQ8BYVkPgHl1DIBtkQwBRa0MgwHJDQGx6QzArgUM=")

const NFFT = 400
const HOP = 160
const NMELS = 80
const NBIN = NFFT / 2 + 1
const AMIN = 1e-10
const TOP_DB = 80

let melW: Float64Array | null = null
let cosT: Float64Array | null = null
let sinT: Float64Array | null = null

function tables(): { mel: Float64Array; cos: Float64Array; sin: Float64Array } {
  if (melW && cosT && sinT) return { mel: melW, cos: cosT, sin: sinT }
  const mel = new Float64Array(NMELS * NBIN)
  for (let m = 0; m < NMELS; m++) {
    const fc = F_CENTRAL[m]
    const bw = BAND[m]
    for (let k = 0; k < NBIN; k++) {
      const f = (k * 8000) / (NBIN - 1)
      const s = (f - fc) / bw
      mel[m * NBIN + k] = Math.max(0, Math.min(s + 1, -s + 1))
    }
  }
  const cos = new Float64Array(NBIN * NFFT)
  const sin = new Float64Array(NBIN * NFFT)
  for (let k = 0; k < NBIN; k++) {
    const th = (2 * Math.PI * k) / NFFT
    for (let n = 0; n < NFFT; n++) {
      cos[k * NFFT + n] = Math.cos(th * n)
      sin[k * NFFT + n] = Math.sin(th * n)
    }
  }
  melW = mel
  cosT = cos
  sinT = sin
  return { mel, cos, sin }
}

export function audioToMel(pcm: Float32Array): Float32Array {
  const { mel, cos, sin } = tables()
  const pad = NFFT / 2
  const xp = new Float64Array(pcm.length + pad * 2)
  for (let i = 0; i < pcm.length; i++) xp[pad + i] = pcm[i]
  const nFrames = Math.floor((pcm.length + pad * 2 - NFFT) / HOP) + 1
  const out = new Float32Array(nFrames * NMELS)
  const re = new Float64Array(NBIN)
  const im = new Float64Array(NBIN)
  for (let f = 0; f < nFrames; f++) {
    const base = f * HOP
    for (let k = 0; k < NBIN; k++) {
      let r = 0
      let q = 0
      const cb = k * NFFT
      for (let n = 0; n < NFFT; n++) {
        const v = xp[base + n] * WIN[n]
        r += v * cos[cb + n]
        q += v * sin[cb + n]
      }
      re[k] = r
      im[k] = q
    }
    const ob = f * NMELS
    for (let m = 0; m < NMELS; m++) {
      let e = 0
      const mb = m * NBIN
      for (let k = 0; k < NBIN; k++) e += (re[k] * re[k] + im[k] * im[k]) * mel[mb + k]
      out[ob + m] = 10 * Math.log10(Math.max(e, AMIN))
    }
  }
  let mx = -Infinity
  for (let i = 0; i < out.length; i++) if (out[i] > mx) mx = out[i]
  const floor = mx - TOP_DB
  for (let i = 0; i < out.length; i++) if (out[i] < floor) out[i] = floor
  return out
}
