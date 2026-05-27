---
name: scope-contract
description: Generate per-task scope contracts with allowed/forbidden globs, acceptance criteria, and rollback plan, plus a CI-ready glob-aware checker that runs on every agent diff.
version: 1.0.0
phase: 14
lesson: 36
tags: [scope, contract, globs, diff-check, ci]
---

بالنظر إلى وصف المهمة وتخطيط الريبو، قم بإنتاج عقد نطاق ومدقق مدرك للفرق.
ينتج:
1. `scope_contract.json` للمهمة التي تحتوي على الحقول: `task_id`، `goal`، `allowed_files` (كرة أرضية)، `forbidden_files` (كرة أرضية)، `acceptance_criteria`، `rollback_plan`، `approvals_required`.
2. `tools/scope_check.py` الذي يأخذ مسار العقد وقائمة الملفات التي تم لمسها ويعيد `ScopeReport` بالإضافة إلى خروج غير صفري على أي مخالفة.
3. الخطوة CI (`.github/workflows/scope-check.yml` أو ما يعادلها) التي تقوم بتشغيل المدقق مقابل فرق الدمج.
4. `outputs/scope/closed/<task_id>.json` اتفاقية أرشيفية بحيث يتم شحن العقود مع سجل التغيير.
الرفض الصارم:
- عقد بدون `forbidden_files`. المساحة السلبية هي جزء من العقد.
- عقد يسرد المسارات الأولية بدلاً من الكرات لأدلة التعليمات البرمجية. تقوم عوامل إعادة البناء بإبطال المسارات الأولية بين عشية وضحاها.
- حقل `rollback_plan` فارغ أو "راجع دليل التشغيل". تهجئتها.
- الموافقات مدرجة على أنها "كل حالة على حدة". يجب أن تكون حدود الموافقة معدودة.
قواعد الرفض:
- إذا كان وصف المهمة لا يقيد منطقة من الريبو، فارفض تأليف `allowed_files` من الوصف وحده. اطلب الدليل الذي تعيش فيه المهمة.
- إذا لم يكن لدى الريبو أمر اختبار، ارفض إضافة `acceptance_criteria` حتى يتم توفيره أو إيقافه. العقد الذي لا يمكن التحقق منه هو رغبة.
- إذا لم يتمكن وقت تشغيل الوكيل من احترام حدود الموافقة (لا يوجد إنسان في الحلقة)، قم بإظهار الفجوة قبل الشحن؛ سيكون زحف النطاق إلى الإجراءات المطلوبة للموافقة هو الفشل السائد.
هيكل الإخراج:
```
<repo>/
├── scope_contract.json
├── outputs/scope/closed/
│   └── T-XXX.json
├── tools/
│   └── scope_check.py
└── .github/
    └── workflows/
        └── scope-check.yml
```

انتهي بـ "ما يجب قراءته بعد ذلك" مشيرًا إلى:
- الدرس 37 الخاص بملاحظات وقت التشغيل التي تربط الأوامر بالعقد.
- الدرس 38 لبوابة التحقق التي تستهلك تقرير النطاق.
- الدرس 39 للوكيل المراجع الذي يقوم بتدقيق أرشيف العقود المغلقة.