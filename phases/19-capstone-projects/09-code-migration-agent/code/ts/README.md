# Code migration agent dashboard (TypeScript skeleton)

هيكل عظمي متعدد الملفات TypeScript لطبقة لوحة المعلومات لترحيل الكود
وكيل كابستون. يعمل الوكيل (Python) في وضع الحماية؛ يعرض هذا الخادم
التقدم للمشغل.

## Layout

- `src/index.ts` — نقطة الدخول، وتحاكي القراد ويقدم اختياريًا HTTP.
- `src/server.ts` — مسارات Hono لـ `/`، `/dashboard`، `/migrations`، `/migrations/:id`.
- `src/migrations.ts` — حالة الجهاز والبيانات الأولية لكل ملف.
- `src/cost.ts` — عدد الأدوار وتنفيذ الميزانية بالدولار.
- `src/types.ts` — الأنواع المشتركة.
- `tests/*.test.ts` — `node --test` اختبارات النمط عبر `tsx`.

## Install

```bash
npm install
```

## Run

```bash
npm start         # offline: simulate 40 ticks and print rollup
npm run serve     # serve the HTML dashboard on PORT (default 8009)
```

## Verify

```bash
npm run typecheck
npm test
```

## Spec references

- مصدر الدرس: `phases/19-capstone-projects/09-code-migration-agent/docs/en.md`
- الوصفات: [OpenRewrite](https://docs.openrewrite.org), libcst.
