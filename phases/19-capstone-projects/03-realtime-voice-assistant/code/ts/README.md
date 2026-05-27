# Capstone 19/03 — Realtime Voice Assistant (TypeScript)

ملفات متعددة TypeScript تسخير عميل الويب لصوت البث pipeline
الموصوفة في `../docs/en.md`. محاكاة آلة الحالة دون اتصال بالإنترنت بالإضافة إلى البث المباشر
WebSocket خادم مدعوم بالباقة `ws`.

## Layout

```text
src/
  index.ts        entry point; runs two offline sessions, probes the live ws, exits 0
  server.ts       hono /healthz + ws upgrade via WebSocketServer
  orchestrator.ts IDLE -> LISTENING -> WAITING -> THINKING -> SPEAKING with barge-in
  vad.ts          turn-completion scorer + synthetic 20ms-frame generator
  protocol.ts     zod-validated frame envelope (event / summary)
  types.ts        AudioChunk, Metrics, SessionOptions, SessionSummary
tests/
  vad.test.ts
  orchestrator.test.ts
  protocol.test.ts
```

## Run

```bash
npm install
npm start                # runs two offline sessions + ws self-probe, exits 0
npm start -- --serve     # keep ws server up; ctrl-c to stop
npm test                 # node --test runner via tsx
npm run typecheck        # tsc --noEmit
```

يؤكد المسار `npm start` غير التفاعلي على وصول الجلسة النظيفة
`first_audio_out`، تسجل جلسة المشاركة حدث مشاركة واحد على الأقل،
ويتلقى المسبار المباشر WebSocket إطارًا `summary` قبل الإغلاق.
