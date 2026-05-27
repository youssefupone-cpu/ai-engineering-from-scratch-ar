---
name: verification-gate
description: Generate a deterministic verification gate that combines scope, rule, and feedback artifacts into a single verification_report.json per task, plus CI wiring that refuses to merge without a green verdict.
version: 1.0.0
phase: 14
lesson: 38
tags: [verification, gate, deterministic, ci, override-log]
---

نظرًا لمعايير قبول المشروع وعناصر طاولة العمل الحالية، قم بإنتاج بوابة التحقق وتجاوز سجل التدقيق.
ينتج:
1. `tools/verify_agent.py` فضح `verify(task_id, artifacts) -> VerdictReport`. دالة خالصة، حتمية، لا توجد مكالمات LLM.
2. `outputs/verification/<task_id>.json` باعتباره المصدر الوحيد لحقيقة الحكم.
3. `tools/override.py` الذي يُلحق إدخالات التجاوز الموقعة بـ `outputs/verification/overrides.jsonl` (يجب أن يتضمن السبب، وهوية المستخدم، والطابع الزمني، ورمز البحث).
4. CI سير العمل الذي يفشل في `passed: false` ويظهر التقرير في السطر.
5. `docs/verification.md` يسرد كل شيك وخطورته ومصدره وسياسة التجاوز.
الرفض الصارم:
- شيك يستدعي LLM. البوابة هي السباكة الحتمية. LLM الحكم يعود للمراجع.
- مسار التجاوز الذي يمكن للوكيل أن يسلكه دون إدخال موقع. التجاوزات هي للبشر فقط.
- تقرير تحقق يغفل المسارات الأثرية التي استهلكها. يجب أن تكون التقارير قابلة للتدقيق.
- نتائج خطورة الكتلة التي يمكن لسير العمل الرجوع إليها بصمت. يتم تحديد الخطورة في وقت الكتابة، وليس في وقت القراءة.
قواعد الرفض:
- إذا لم يكن للمشروع أمر قبول، ارفض شحن البوابة حتى وجودها. البوابة التي لا تثبت شيئا هي المسرح.
- إذا كان تقرير القاعدة غير موجود، ارفض تخطي فحص القاعدة؛ فشل مغلقة.
- إذا كان سجل الملاحظات غير موجود، فارفض تخطي فحص القبول؛ السجلات المفقودة هي في حد ذاتها كتلة.
- إذا لم تكن إدخالات التجاوز خاضعة للتحكم في الإصدار، فارفض توصيل مسار التجاوز؛ التجاوزات غير الرسمية تهزم البوابة.
هيكل الإخراج:
```
<repo>/
├── tools/
│   ├── verify_agent.py
│   └── override.py
├── outputs/verification/
│   ├── overrides.jsonl
│   └── <task_id>.json
├── docs/verification.md
└── .github/workflows/verify.yml
```

انتهي بـ "ما يجب قراءته بعد ذلك" مشيرًا إلى:
- الدرس 39 للوكيل المراجع الذي ينتزع بعد الحكم الأخضر.
- الدرس 40 لمولد التسليم الذي يتضمن الحكم في الحزمة.
- الدرس 41 لتشغيل البوابة مقابل نموذج تطبيق حقيقي.