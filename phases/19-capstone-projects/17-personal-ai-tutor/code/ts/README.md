# Lesson 17 - Personal AI Tutor (TypeScript web app)

TypeScriptنصف الكابونة. يشحن جانب بايثون نموذج المتعلم و
سياسة المعلم؛ يعرض هذا المشروع سطح تطبيق الويب: المنهج الدراسي DAG
ووكر، نموذج التعلم بأسلوب BKT، والتكرار المتباعد FSRS
جدولة خلف طريقين HTTP.

## Layout

```text
src/
  index.ts       entry: demo (default) or HTTP server (--serve)
  server.ts      Hono routes (GET /lesson/next, POST /lesson/:id/submit)
  curriculum.ts  DAG fixture + Kahn topo sort + next-lesson picker
  mastery.ts     MasteryStore (per-lesson BKT-ish update)
  repetition.ts  scheduleNextDue (interval doubling / halving, clamped)
  types.ts       Lesson, Mastery, Pick
tests/
  curriculum.test.ts  topo order, BKT update, FSRS scheduling
```

## Run

```bash
npm install
npm run typecheck
npm test
npm start            # self-terminating curriculum walk
npm run serve        # HTTP server on :8090
```
