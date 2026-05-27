# Multi-agent software team (TypeScript skeleton)

هيكل عظمي متعدد الملفات TypeScript لفريق البرمجيات متعدد الوكلاء.
يتشارك وكلاء المخططين والمبرمجين والمراجعين في مساحة عمل ويتناوبون من خلال
منسق. يقوم كعب شجرة العمل بتشغيل العمليات الفرعية عبر execFile مع ملف
قائمة الحظر ورفض Shell-metachar.

## Layout

- `src/index.ts` — عداء تجريبي.
- `src/agent.ts` — القاعدة `Agent` فئة زائد `PlannerAgent`، `CoderAgent`، `ReviewerAgent`.
- `src/coordinator.ts` — حلقة دائرية وتتبع الدوران.
- `src/workspace.ts` — نظام الملفات وسجل الرسائل المشترك في الذاكرة.
- `src/runtime.ts` — `child_process.execFile` كعب شجرة العمل مع قائمة الرفض.
- `src/types.ts` — الأنواع المشتركة.
- `tests/*.test.ts` — `node --test` اختبارات النمط عبر `tsx`.

## Install

```bash
npm install
```

## Run

```bash
npm start
```

## Verify

```bash
npm run typecheck
npm test
```

## Spec references

- مصدر الدرس: `phases/19-capstone-projects/10-multi-agent-software-team/docs/en.md`
- [MetaGPT](https://githubhub.com/FoundationAgents/MetaGPT) إطار عمل متعدد الوكلاء قائم على الأدوار.
