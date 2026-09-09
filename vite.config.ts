import fs from 'node:fs'
import path from 'node:path'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// HTTPS 마이크(getUserMedia)용 인증서. 기본은 이 개발 머신의 공유 인증서이고,
// CERT_DIR 환경변수로 바꿀 수 있다. 키는 저장소에 커밋하지 않는다.
const CERT_DIR = process.env.CERT_DIR ?? '/data/workspace/ai2breath/vocal_resonance_annotation/certs'

function httpsOption() {
  if (!process.env.HTTPS) return undefined
  const key = path.join(CERT_DIR, 'key.pem')
  const cert = path.join(CERT_DIR, 'cert.pem')
  if (!fs.existsSync(key) || !fs.existsSync(cert)) {
    throw new Error(`인증서가 없습니다: ${CERT_DIR}/{key,cert}.pem`)
  }
  return { key: fs.readFileSync(key), cert: fs.readFileSync(cert) }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ['onnxruntime-web'],
  },
  server: {
    allowedHosts: ['chr747.iptime.org', 'localhost'],
    https: httpsOption(),
    proxy: {
      // ai2breath-train의 실시간 임베딩 서버 (ws://127.0.0.1:8765)
      '/ws/live': {
        target: 'ws://127.0.0.1:8765',
        ws: true,
        rewrite: (p) => p.replace(/^\/ws\/live/, '/'),
      },
    },
  },
})
