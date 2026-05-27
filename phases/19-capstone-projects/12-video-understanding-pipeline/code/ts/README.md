# Lesson 12 - Video Understanding Pipeline (TypeScript UI)

TypeScriptنصف الكابونة. يمتلك جانب بايثون (`code/main.py`) ملف
مؤشر متعدد المتجهات والتأريض الزمني. يقوم هذا المشروع بشحن لوحة القيادة
النصف: تطبيق Hono على مراحل pipeline الأربع (قطعة، تضمين، فهرس، qa).

## Layout

```text
src/
  index.ts     entry: demo (default) or HTTP server (--serve)
  server.ts    Hono routes (/, /jobs, /job/:id) + HTML index
  jobs.ts     JobStore + fixture seeder
  stages.ts    stage advance + overall status
  types.ts     Stage, StageState, Job
tests/
  stages.test.ts  job state transitions + store
```

## Run

```bash
npm install
npm run typecheck
npm test
npm start              # self-terminating demo
npm run serve          # HTTP server on :8123
```
