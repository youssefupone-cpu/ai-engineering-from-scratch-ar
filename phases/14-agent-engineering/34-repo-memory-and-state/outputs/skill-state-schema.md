---
name: state-schema
description: Generate project-specific JSON Schemas for agent state and task board, a Python StateManager with atomic writes, and a migration scaffold so schema bumps cannot corrupt the workbench.
version: 1.0.0
phase: 14
lesson: 34
tags: [state, schema, json-schema, atomic-writes, migrations]
---

بالنظر إلى الريبو ومنتج الوكيل الذي يعمل بداخله، قم بإنتاج ملفات حالة المخطط الأول لمنضدة العمل.
ينتج:
1. `schemas/agent_state.schema.json` يغطي المفاتيح المطلوبة، وقيم الحالة المسموح بها، ونظام المصفوفة مقابل فارغة، وعدد صحيح `schema_version`.
2. `schemas/task_board.schema.json` يغطي نمط معرف المهمة، والمالكين المسموح بهم، والحالات المسموح بها، ومصفوفات القبول.
3. `tools/state_manager.py` تعريض `load`، `commit`، و`update` مع الكتابة الذرية المؤقتة وإعادة التسمية.
4. `tools/migrate_state.py` سقالة لمخطط المخطط التالي، بصوت عالٍ إذا كان الملف من إصدار غير معروف.
5. `agent_state.json` و`task_board.json` مصنفان في `schema_version: 1` وتراكم جديد.
الرفض الصارم:
- مخطط بدون حقل `schema_version`. الهجرات ليست اختيارية
- السماح بـ `null` حيث من المتوقع وجود مصفوفة. `null` هو خطأ أثناء الكتابة يتنكر في هيئة بيانات.
- كاتب يستخدم `open(path, "w")` عادي. الذري يكتب فقط؛ الملفات الجزئية تفسد مصدر الحقيقة.
- تخزين الرموز المميزة أو نصوص الدردشة الأولية أو PII داخل الحالة. الدولة هي للحقائق ذات الصلة الريبو.
قواعد الرفض:
- إذا لم يكن لدى الريبو تحكم في الإصدار، ارفض شحن ملفات الحالة. Atomic يكتب بالإضافة إلى git diff هي قصة المتانة.
- إذا لم يكن لدى المشروع أمر قبول واحد على الأقل للتحقق من صحة انتقال `done`، فارفض قيمة التعداد `status: done`. إن إضافة `done` بدون التحقق من القبول يعد بمثابة مسرحية.
- إذا كان المشروع ينوي مشاركة الحالة عبر العمليات دون استراتيجية قفل، فاكشف عن هذه النتيجة قبل الشحن؛ إعادة التسمية الذرية ضرورية ولكنها ليست كافية.
هيكل الإخراج:
```
<repo>/
├── agent_state.json
├── task_board.json
├── schemas/
│   ├── agent_state.schema.json
│   └── task_board.schema.json
└── tools/
    ├── state_manager.py
    └── migrate_state.py
```

انتهي بـ "ما يجب قراءته بعد ذلك" مشيرًا إلى:
- الدرس 35 لبرنامج التهيئة الذي يستدعي المدير عند بدء التشغيل.
- الدرس 38 لبوابة التحقق التي تقرأ الحالة لدرجة الإكتمال.
- الدرس 40 لمولد التسليم الذي يستهلك نفس المخطط.