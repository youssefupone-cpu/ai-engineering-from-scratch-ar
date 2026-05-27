# LLM observability dashboard (TypeScript skeleton)

هيكل عظمي متعدد الملفات TypeScript لغطاء لوحة القيادة LLM القابلية للملاحظة.
يقبل خادم Hono امتدادات OpenTelemetry GenAI، ويحتفظ بها في حلقة 10 كيلو
المخزن المؤقت، ويعرض زمن الوصول p50/p95/p99 والتكلفة لكل نموذج.

## Layout

- `src/index.ts` — نقطة الدخول، البذور الاصطناعية وتخدم اختياريًا HTTP.
- `src/server.ts` — مسارات Hono لـ `/trace`، `/`، `/dashboard`، `/dashboard.json`، `/healthz`.
- `src/spans.ts` — `RingBuffer` و `ObservabilityStore` (امتداد 10 كيلو افتراضيًا).
- `src/rollup.ts` — `percentile` و `rollUpByModel`.
- `src/pricing.ts` — أسعار كل طراز ومساعدي التكلفة لعام 2026.
- `src/types.ts` — الأنواع المشتركة.
- `tests/*.test.ts` — `node --test` اختبارات النمط عبر `tsx`.

## Install

```bash
npm install
```

## Run

```bash
npm start         # seeds 1200 synthetic spans and prints the rollup
npm run serve     # also serves the HTTP ingest + dashboard on PORT (default 8011)
```

## Verify

```bash
npm run typecheck
npm test
```

## Spec references

- مصدر الدرس: `phases/19-capstone-projects/11-llm-observability-dashboard/docs/en.md`
- [الاصطلاحات الدلالية OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
