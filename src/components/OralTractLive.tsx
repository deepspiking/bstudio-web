/**
 * 언어 투영 — ecapa LDA 축(+/−) 실시간 2D 탭.
 *
 * 위: 배경(업로드 vocal-tract.svg) + 라이브 펄스(+ 파랑/− 빨강 그라데이션).
 * 아래: 녹음될 때마다 한 줄씩 쌓이는 "파형 + 언어신호" 줄. 파형 위에 x값을
 * 세로축(위 + / 아래 −)으로 겹쳐 그린다. 신호는 VAD(발화) 구간만 이어 그리고,
 * 그 외 구간은 보간하지 않는다. 줄을 클릭하면 그 위치부터 재생된다.
 */
import { useCallback, useEffect, useRef, useState } from 'react'

interface Frame {
  t: number
  x: number
  y: number
}
interface Capture {
  audio: Float32Array
  frames: Frame[]
  duration: number
}

const SAMPLE_RATE = 16000
const VIEW = 'xvector/lang2d_logit'
const ENCODER = 'xvector'
const XMIN = -1.12
const XMAX = 1.12
const PAD_X = 52
const PAD_Y = 14
const FIG_W = 250
const FIG_H = 580
const FIG_X0 = 12
const FIG_Y0 = 12
// 거울 변환까지 적용한 vocal-tract.svg 안 구강 중심 — 혓바닥 바로 위 공동 위치 (파일 좌표)
const MOUTH_X = 176
const MOUTH_Y = 150
// 배경 그림 크기(폭 기준 비율) — 1보다 작으면 좌우 여백이 생기며 축소된다
const FIG_SCALE = 0.5
// EPD(발성/무음 판정) 게이트 — 이 dBFS 이상의 프레임만 발성으로 본다
const EPD_DBFS = -50
const PULSE_R = 30
// x축(+/-) 표시 폭 — 플롯 폭 대비. 작을수록 축이 좁아진다
const AXIS_FRAC = 0.62
const HOP_MS = 100
const SILENCE_MS = 500
const MIN_SPEECH_SEC = 0.4
const PREROLL_SEC = 0.15
const KEEP_CHUNKS = Math.ceil(30000 / HOP_MS)
// 프레임 간 이 간격보다 크면 VAD가 끊긴 구간 — 선을 잇지 않는다
const GAP_SEC = 0.18

const MIC_WORKLET = `
class MicProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const chunk = (options && options.processorOptions && options.processorOptions.chunk) || 1600;
    this.buf = new Float32Array(chunk); this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.n++] = ch[i];
      if (this.n === this.buf.length) {
        const out = new Int16Array(this.buf.length);
        for (let k = 0; k < out.length; k++) {
          const v = Math.max(-1, Math.min(1, this.buf[k]));
          out[k] = v * 32767;
        }
        this.port.postMessage(out.buffer, [out.buffer]);
        this.n = 0;
      }
    }
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);
`

/** VAD 끊김(GAP_SEC 초과)마다 잘라 발화 연속 구간들로 나눈다 */
function runsOf(frames: Frame[]): Frame[][] {
  const runs: Frame[][] = []
  let cur: Frame[] = []
  for (const f of frames) {
    if (cur.length && f.t - cur[cur.length - 1].t > GAP_SEC) {
      if (cur.length >= 2) runs.push(cur)
      cur = []
    }
    cur.push(f)
  }
  if (cur.length >= 2) runs.push(cur)
  return runs
}

function hueOf(v: number): number {
  return 210 - Math.max(-1, Math.min(1, v)) * 105 // −1=빨강(0) → +1=파랑(210)
}

/** 공명 표시 — 퍼지는 그라데이션 동심원. +일수록 파랑, −일수록 빨강 */
function drawPulse(ctx: CanvasRenderingContext2D, x: number, y: number, v: number, now: number) {
  const h = hueOf(v)
  const breathe = 1 + 0.08 * Math.sin(now / 420)
  const R = PULSE_R * breathe
  ctx.save()
  ctx.shadowColor = `hsl(${h}, 78%, 62%)`
  ctx.shadowBlur = 10
  const g = ctx.createRadialGradient(x, y, 0, x, y, R)
  g.addColorStop(0, `hsla(${h}, 85%, 66%, 0.5)`)
  g.addColorStop(0.5, `hsla(${h}, 80%, 60%, 0.22)`)
  g.addColorStop(1, `hsla(${h}, 75%, 58%, 0)`)
  ctx.fillStyle = g
  ctx.beginPath()
  ctx.arc(x, y, R, 0, Math.PI * 2)
  ctx.fill()
  ctx.shadowBlur = 0
  for (const k of [0.3, 0.58, 0.84]) {
    ctx.strokeStyle = `hsla(${h}, 80%, 66%, ${0.55 - k * 0.4})`
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.arc(x, y, R * k, 0, Math.PI * 2)
    ctx.stroke()
  }
  ctx.restore()
}

/** 발화 하나의 가로줄 — 회색 파형 + 언어신호(VAD 구간만, + 위/− 아래). 클릭=그 위치부터 재생 */
function RowWave({ cap, active, onSeek }: {
  cap: Capture
  active: boolean
  onSeek: (t: number) => void
}) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const cvRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const wrap = wrapRef.current
    const cv = cvRef.current
    if (!wrap || !cv) return
    const drawRow = () => {
      const w = wrap.clientWidth
      const h = wrap.clientHeight
      if (!w || !h) return
      const dpr = window.devicePixelRatio || 1
      if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
        cv.width = Math.round(w * dpr)
        cv.height = Math.round(h * dpr)
      }
      const ctx = cv.getContext('2d')
      if (!ctx) return
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, w, h)

      const labelW = 34
      const pw = w - labelW
      const midY = h / 2
      const amp = h * 0.42
      const tx = (t: number) => labelW + (t / Math.max(cap.duration, 1e-6)) * pw

      ctx.fillStyle = '#697083'
      ctx.font = '12px system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('+', labelW - 16, 10)
      ctx.fillText('−', labelW - 16, h - 6)

      const n = cap.audio.length
      ctx.strokeStyle = '#4a515f'
      ctx.lineWidth = 1
      ctx.beginPath()
      if (n > 0) {
        for (let px = 0; px < pw; px++) {
          const s0 = Math.floor((px / pw) * n)
          const s1 = Math.max(s0 + 1, Math.floor(((px + 1) / pw) * n))
          let mn = 1
          let mx = -1
          for (let i = s0; i < s1; i++) {
            const val = cap.audio[i]
            if (val < mn) mn = val
            if (val > mx) mx = val
          }
          ctx.moveTo(tx(0) + px + 0.5, midY - mx * amp)
          ctx.lineTo(tx(0) + px + 0.5, midY - mn * amp)
        }
      }
      ctx.stroke()

      const sy = (x: number) =>
        Math.max(6, Math.min(h - 6, midY - (Math.max(-1.1, Math.min(1.1, x)) / 1.1) * (midY - 6)))
      ctx.lineWidth = 2
      ctx.lineJoin = 'round'
      for (const run of runsOf(cap.frames)) {
        let px0 = tx(run[0].t)
        for (let i = 1; i < run.length; i++) {
          const a = run[i - 1]
          const b = run[i]
          const x0 = i === 1 ? px0 : labelW + (a.t / Math.max(cap.duration, 1e-6)) * pw
          px0 = labelW + (a.t / Math.max(cap.duration, 1e-6)) * pw
          ctx.beginPath()
          ctx.moveTo(x0, sy(a.x))
          ctx.lineTo(labelW + (b.t / Math.max(cap.duration, 1e-6)) * pw, sy(b.x))
          ctx.strokeStyle = `hsl(${hueOf((a.x + b.x) / 2)}, 82%, 62%)`
          ctx.stroke()
        }
      }
      ctx.globalAlpha = 1
    }
    drawRow()
    const ro = new ResizeObserver(drawRow)
    ro.observe(wrap)
    return () => ro.disconnect()
  }, [cap])

  return (
    <div
      ref={wrapRef}
      className="embed-panel"
      onClick={(e) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const labelW = 34
        const pw = Math.max(1, rect.width - labelW)
        const t = Math.max(0, Math.min(cap.duration, ((e.clientX - rect.left - labelW) / pw) * cap.duration))
        onSeek(t)
      }}
      style={{
        position: 'relative',
        height: 64,
        padding: 0,
        marginBottom: 6,
        overflow: 'hidden',
        cursor: 'pointer',
        outline: active ? '1px solid #7d8bff' : '1px solid transparent',
      }}
    >
      <canvas ref={cvRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%' }} />
    </div>
  )
}

export default function OralTractLive() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const [phase, setPhase] = useState<'idle' | 'live' | 'captured'>('idle')
  const [listLen, setListLen] = useState(0)
  const [active, setActive] = useState(-1)
  const [playTime, setPlayTime] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [viewKey] = useState<string>(VIEW)
  const [imgBox, setImgBox] = useState<{ left: number; top: number; width: number; height: number } | null>(null)

  const listRef = useRef<Capture[]>([])
  const activeRef = useRef<Capture | null>(null)
  const playingIdxRef = useRef(-1)
  const playheadRef = useRef<number | null>(null)
  const viewKeyRef = useRef(viewKey)
  viewKeyRef.current = viewKey
  const encoderRef = useRef(ENCODER)
  encoderRef.current = ENCODER
  const liveRef = useRef({ x: 0, y: 0, on: false, trail: [] as Frame[] })
  const capRef = useRef<{ pcm: Int16Array[]; base: number; frames: Frame[]; idleRun: number; speech: number }>({
    pcm: [],
    base: 1,
    frames: [],
    idleRun: 0,
    speech: 0,
  })
  const wsRef = useRef<WebSocket | null>(null)
  const micCtxRef = useRef<AudioContext | null>(null)
  const playCtxRef = useRef<AudioContext | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const rafRef = useRef(0)
  const playRafRef = useRef(0)
  const playSrcRef = useRef<AudioBufferSourceNode | null>(null)

  // 투영 공간이 바뀌면 이전 공간의 점·기록은 의미가 없어 비운다
  useEffect(() => {
    liveRef.current = { x: 0, y: 0, on: false, trail: [] }
    capRef.current.frames = []
    listRef.current = []
    setListLen(0)
    setActive(-1)
    activeRef.current = null
  }, [])
  const phaseRef = useRef(phase)
  phaseRef.current = phase

  useEffect(() => {
    const el = panelRef.current
    if (!el) return
    const layout = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      if (!w || !h) return
      const plotW = Math.max(120, w - PAD_X * 2)
      const bw = Math.max(90, plotW * FIG_SCALE)
      const s = bw / FIG_W
      const ph = Math.max(60, h - PAD_Y * 2)
      const zeroPx = PAD_X + 0.5 * plotW
      const zeroY = PAD_Y + 0.5 * ph
      setImgBox({
        // 구강 중심이 x축 0(축 중심)과 캔버스 세로 중앙에 오도록 배치
        left: Math.round(zeroPx - (MOUTH_X - FIG_X0) * s),
        top: Math.round(zeroY - (MOUTH_Y - FIG_Y0) * s),
        width: Math.round(bw),
        height: Math.round(FIG_H * s),
      })
    }
    layout()
    const ro = new ResizeObserver(layout)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const draw = useCallback(() => {
    const cv = canvasRef.current
    if (!cv) return
    const ctx = cv.getContext('2d')
    if (!ctx) return
    const dpr = window.devicePixelRatio || 1
    const w = cv.clientWidth
    const h = cv.clientHeight
    if (w === 0 || h === 0) return
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr)
      cv.height = Math.round(h * dpr)
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, w, h)
    const now = performance.now()

    const padY = PAD_Y
    const pw = w - PAD_X * 2
    const ph = h - padY * 2
    const axisSpan = pw * AXIS_FRAC
    const axisLeft = PAD_X + (pw - axisSpan) / 2
    const clampX = (x: number) => Math.max(-1, Math.min(1, x))
    const sx = (x: number) => axisLeft + ((clampX(x) - XMIN) / (XMAX - XMIN)) * axisSpan
    const sy = (y: number) => padY + ((y + 0.5) / 1.0) * ph
    const zeroY = sy(0)

    ctx.strokeStyle = '#3a3f4b'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(sx(-1), zeroY)
    ctx.lineTo(sx(1), zeroY)
    ctx.stroke()
    ctx.fillStyle = '#8b93a3'
    ctx.font = '11px system-ui, sans-serif'
    ctx.textAlign = 'center'
    for (const t of [-1, -0.5, 0, 0.5, 1]) {
      const X = sx(t)
      ctx.beginPath()
      ctx.moveTo(X, zeroY - 4)
      ctx.lineTo(X, zeroY + 4)
      ctx.stroke()
      ctx.fillText(t.toFixed(1), X, zeroY + 18)
    }

    if (phase === 'live' && liveRef.current.on) {
      const live = liveRef.current
      ctx.globalAlpha = 0.28
      ctx.strokeStyle = '#c3cad8'
      ctx.lineWidth = 1
      for (const run of runsOf(live.trail)) {
        ctx.beginPath()
        run.forEach((f, i) => {
          const X = sx(f.x)
          const Y = sy(f.y)
          if (i === 0) ctx.moveTo(X, Y)
          else ctx.lineTo(X, Y)
        })
        ctx.stroke()
      }
      ctx.globalAlpha = 1
      drawPulse(ctx, sx(live.x), sy(live.y), live.x, now)
    }

    const cap = activeRef.current
    if (phase === 'captured' && cap) {
      ctx.globalAlpha = 0.4
      ctx.lineWidth = 1.3
      for (const run of runsOf(cap.frames)) {
        ctx.beginPath()
        run.forEach((f, i) => {
          const X = sx(f.x)
          const Y = sy(f.y)
          if (i === 0) ctx.moveTo(X, Y)
          else ctx.lineTo(X, Y)
        })
        ctx.strokeStyle = '#f5f7fa'
        ctx.stroke()
      }
      ctx.globalAlpha = 1
      const head = playheadRef.current
      if (head != null && cap.frames.length) {
        // 보간하지 않음 — 마지막 발화 프레임 값을 유지(VAD 밖에선 안 움직인다)
        let pos = cap.frames[0]
        for (const f of cap.frames) {
          if (f.t <= head) pos = f
          else break
        }
        drawPulse(ctx, sx(pos.x), sy(pos.y), pos.x, now)
      }
    }
  }, [phase])

  useEffect(() => {
    const loop = () => {
      draw()
      rafRef.current = requestAnimationFrame(loop)
    }
    rafRef.current = requestAnimationFrame(loop)
    const onResize = () => draw()
    window.addEventListener('resize', onResize)
    return () => {
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', onResize)
    }
  }, [draw])

  const stopPlayback = useCallback(() => {
    cancelAnimationFrame(playRafRef.current)
    playSrcRef.current?.stop()
    playSrcRef.current = null
    playheadRef.current = null
    playingIdxRef.current = -1
  }, [])

  const finishCapture = useCallback(() => {
    const cap = capRef.current
    const total = cap.pcm.reduce((s, c) => s + c.length, 0)
    if (total === 0) return
    const audio = new Float32Array(total)
    let o = 0
    for (const c of cap.pcm) {
      for (let i = 0; i < c.length; i++) audio[o + i] = c[i] / 32768
      o += c.length
    }
    const step = Math.max(1, Math.round((HOP_MS / 1000) * SAMPLE_RATE))
    const thresh = 10 ** (EPD_DBFS / 20)
    let firstLoud = -1
    for (let s = 0; s + step <= audio.length; s += step) {
      let sum = 0
      for (let i = s; i < s + step; i++) sum += audio[i] * audio[i]
      if (Math.sqrt(sum / step) >= thresh) {
        firstLoud = s
        break
      }
    }
    const start = firstLoud < 0 ? 0 : Math.max(0, firstLoud - Math.round(PREROLL_SEC * SAMPLE_RATE))
    const trimmed = start > 0 ? audio.subarray(start) : audio
    const shift = start / SAMPLE_RATE
    const out: Capture = {
      audio: trimmed,
      frames: cap.frames.map((f) => ({ ...f, t: f.t - shift })).filter((f) => f.t >= 0),
      duration: trimmed.length / SAMPLE_RATE,
    }
    const idx = listRef.current.length
    listRef.current = [...listRef.current, out]
    setListLen(idx + 1)
    setActive(idx)
    activeRef.current = out
  }, [])

  const stopLive = useCallback(
    (keep: boolean) => {
      wsRef.current?.close()
      wsRef.current = null
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
      void micCtxRef.current?.close()
      micCtxRef.current = null
      liveRef.current = { x: 0, y: 0, on: false, trail: [] }
      if (keep) finishCapture()
      setPhase(keep ? 'captured' : 'idle')
    },
    [finishCapture],
  )

  const playFrom = useCallback(
    (idx: number, fromSec: number) => {
      const cap = listRef.current[idx]
      if (!cap) return
      stopPlayback()
      const ctx = playCtxRef.current ?? new AudioContext()
      playCtxRef.current = ctx
      void ctx.resume()
      const buf = ctx.createBuffer(1, cap.audio.length, SAMPLE_RATE)
      buf.getChannelData(0).set(cap.audio)
      const src = ctx.createBufferSource()
      src.buffer = buf
      src.connect(ctx.destination)
      playSrcRef.current = src
      const t0 = ctx.currentTime
      const from = Math.max(0, Math.min(cap.duration - 0.02, fromSec))
      src.start(0, from)
      playingIdxRef.current = idx
      activeRef.current = cap
      setActive(idx)
      setPlayTime(from)
      playheadRef.current = from
      const tick = () => {
        const t = from + (ctx.currentTime - t0)
        if (t >= cap.duration) {
          stopPlayback()
          return
        }
        playheadRef.current = t
        setPlayTime(t)
        playRafRef.current = requestAnimationFrame(tick)
      }
      playRafRef.current = requestAnimationFrame(tick)
    },
    [stopPlayback],
  )

  const startLive = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        window.isSecureContext
          ? '이 브라우저는 마이크 입력을 지원하지 않습니다.'
          : `마이크는 HTTPS 또는 localhost에서만 쓸 수 있습니다 (현재 ${location.protocol}//${location.host}).`,
      )
      return
    }
    stopPlayback()
    stopLive(false)
    capRef.current = { pcm: [], base: 1, frames: [], idleRun: 0, speech: 0 }
    liveRef.current = { x: 0, y: 0, on: false, trail: [] }
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: false,
        },
      })
      streamRef.current = stream
      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
      micCtxRef.current = ctx
      const url = URL.createObjectURL(new Blob([MIC_WORKLET], { type: 'text/javascript' }))
      await ctx.audioWorklet.addModule(url)
      URL.revokeObjectURL(url)

      const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`)
      ws.binaryType = 'arraybuffer'
      wsRef.current = ws
      await new Promise<void>((resolve, reject) => {
        ws.onopen = () => resolve()
        ws.onerror = () => reject(new Error('연결 실패 — live 서버가 떠 있는지 확인하세요'))
      })
      ws.send(
        JSON.stringify({
          encoder: encoderRef.current,
          windowSec: 1,
          hopMs: HOP_MS,
          gateDbfs: EPD_DBFS,
          minWindowSpeech: 0.5,
        }),
      )
      setPhase('live')

      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data as string)
        if (msg.type === 'ready') return
        const cap = capRef.current
        const quiet = (msg.chunk_dbfs ?? msg.rms_dbfs ?? 0) < EPD_DBFS
        if (msg.type === 'point') {
          const c = msg.coords?.[viewKeyRef.current]
          if (c) {
            const f: Frame = { t: (msg.chunk ?? cap.frames.length) * (HOP_MS / 1000), x: c[0], y: 0 }
            cap.frames.push(f)
            liveRef.current.x = c[0]
            liveRef.current.y = 0
            liveRef.current.on = true
            liveRef.current.trail.push(f)
            if (liveRef.current.trail.length > 60) liveRef.current.trail.shift()
          }
        }
        if (quiet) cap.idleRun += 1
        else {
          cap.idleRun = 0
          cap.speech += 1
        }
        const speechOk = cap.speech * (HOP_MS / 1000) >= MIN_SPEECH_SEC
        const idleOk = cap.idleRun * (HOP_MS / 1000) >= SILENCE_MS / 1000
        if (speechOk && idleOk && phaseRef.current === 'live') stopLive(true)
      }
      ws.onclose = () => {
        if (phaseRef.current === 'live') stopLive(false)
      }

      const src = ctx.createMediaStreamSource(stream)
      const node = new AudioWorkletNode(ctx, 'mic-processor', {
        processorOptions: { chunk: Math.round((HOP_MS / 1000) * SAMPLE_RATE) },
      })
      node.port.onmessage = (e) => {
        const buf = e.data as ArrayBuffer
        const cap = capRef.current
        cap.pcm.push(new Int16Array(buf.slice(0)))
        if (cap.pcm.length > KEEP_CHUNKS) {
          cap.base += cap.pcm.length - KEEP_CHUNKS
          cap.pcm.splice(0, cap.pcm.length - KEEP_CHUNKS)
        }
        if (ws.readyState === WebSocket.OPEN) ws.send(buf)
      }
      src.connect(node)
      node.connect(ctx.createGain()).connect(ctx.destination)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      stopLive(false)
    }
  }, [stopLive, stopPlayback])

  useEffect(
    () => () => {
      cancelAnimationFrame(rafRef.current)
      cancelAnimationFrame(playRafRef.current)
      wsRef.current?.close()
      streamRef.current?.getTracks().forEach((t) => t.stop())
      void micCtxRef.current?.close()
      void playCtxRef.current?.close()
    },
    [],
  )

  const activeCap = activeRef.current

  return (
    <div className="embed-wrap">
      <div
        ref={panelRef}
        className="embed-panel"
        style={{
          position: 'relative',
          overflow: 'hidden',
          padding: 0,
          height: 'min(74vh, 680px)',
          minHeight: 420,
        }}
      >
        {imgBox && (
          <img
            src="/vocal-tract.svg"
            alt=""
            aria-hidden="true"
            style={{
              position: 'absolute',
              left: imgBox.left,
              top: imgBox.top,
              width: imgBox.width,
              height: imgBox.height,
              opacity: 0.16,
              pointerEvents: 'none',
            }}
          />
        )}
        <canvas
          ref={canvasRef}
          style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', display: 'block' }}
        />
        {error && (
          <p className="hint" style={{ position: 'absolute', left: 12, top: 8, color: '#eb5757' }}>
            {error}
          </p>
        )}
        {phase !== 'live' ? (
          <button
            className="btn primary"
            style={{
              position: 'absolute',
              left: '50%',
              top: '50%',
              transform: 'translate(-50%, -50%)',
              padding: '14px 26px',
              fontSize: 16,
              zIndex: 2,
            }}
            onClick={() => void startLive()}
          >
            {listLen > 0 ? '새로 녹음' : '듣기 시작'}
          </button>
        ) : (
          <>
            <button
              className="btn small"
              style={{
                position: 'absolute',
                left: '50%',
                top: '50%',
                transform: 'translate(-50%, -50%)',
                zIndex: 2,
              }}
              onClick={() => stopLive(true)}
            >
              정지
            </button>
            <p className="hint" style={{ position: 'absolute', left: 12, top: 8 }}>
              듣는 중 — 말을 마치고 잠시 쉬면 자동 저장됩니다
            </p>
          </>
        )}
      </div>

      <div style={{ marginTop: 8 }}>
        {listRef.current.map((cap, i) => (
          <div key={i} style={{ position: 'relative' }}>
            <span
              className="hint"
              style={{
                position: 'absolute',
                left: 6,
                top: 2,
                zIndex: 2,
                pointerEvents: 'none',
                fontSize: 10,
                color: i === active ? '#b7c3ff' : '#6a7180',
              }}
            >
              #{i + 1} · {cap.duration.toFixed(2)}s
            </span>
            <RowWave cap={cap} active={i === active} onSeek={(t) => playFrom(i, t)} />
          </div>
        ))}
        {playingIdxRef.current >= 0 && (
          <p className="hint" style={{ textAlign: 'center', margin: '2px 0 0' }}>
            {playingIdxRef.current + 1}번 발화 재생 중 {playTime.toFixed(2)}s —{' '}
            <span
              style={{ cursor: 'pointer', textDecoration: 'underline' }}
              onClick={() => stopPlayback()}
              role="button"
            >
              정지
            </span>
          </p>
        )}
        {activeCap && activeCap.frames.length > 0 && playingIdxRef.current < 0 && (
          <p className="hint" style={{ textAlign: 'center', margin: '2px 0 0' }}>
            {active}번 발화 — 파형 줄을 클릭하면 그 위치부터 재생
          </p>
        )}
      </div>
    </div>
  )
}
