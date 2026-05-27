# Lesson 16 - GitHub Issue-to-PR Agent (TypeScript webhook receiver)

TypeScriptنصف الكابونة. يقوم جانب بايثون بشحن حلقة الوكيل و
المرسل؛ YAML يشحن الجانب سير عمل الإجراءات. هذا المشروع هو GitHub
جهاز استقبال الخطاف على الويب للتطبيق: HMAC التحقق من الجسم الخام، والتوجيه على نوع الحدث، والإرسال
وكيل كعب لـ `issues.opened`.

## Layout

```text
src/
  index.ts    entry: demo (default) or HTTP server (--serve)
  server.ts   Hono webhook receiver (POST /webhook)
  verify.ts   X-Hub-Signature-256 HMAC, timing-safe
  router.ts   event-type routing (ping, issues, pull_request)
  agent.ts    stub agent + audit log
  types.ts    payload + audit shapes
tests/
  verify.test.ts  signature pass, tampered, router pathing
```

## Run

```bash
npm install
npm run typecheck
npm test
npm start            # self-terminating demo (in-process replays)
npm run serve        # HTTP server on :8081
```

تتم قراءة السر HMAC من `GH_WEBHOOK_SECRET` (الافتراضي `demo-shared-secret`
للعرض).
