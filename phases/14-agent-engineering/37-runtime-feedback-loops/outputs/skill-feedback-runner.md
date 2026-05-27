---
name: feedback-runner
description: Wrap shell commands with deterministic stdout/stderr/exit/duration capture, persist a JSONL record per command, and refuse to advance the agent loop when feedback is missing.
version: 1.0.0
phase: 14
lesson: 37
tags: [feedback, subprocess, runner, jsonl, loop-control]
---

بالنظر إلى مشروع يقوم بتشغيل أوامر shell داخل حلقة الوكيل، قم بإنتاج مشغل ردود الفعل وJSONL الذي يكتبه.

Produce:

1. `tools/run_with_feedback.py` تعريض `run_with_feedback(command: list[str], agent_note: str, timeout_s: float) -> FeedbackRecord`.
2. `feedback_record.jsonl` الموقع أسفل طاولة العمل، سجل واحد في كل سطر.
3. `tools/feedback_loader.py` يقوم بإرجاع أحدث سجلات N للمهمة النشطة.
4. يتم استدعاء المساعد `loop_can_advance(record) -> bool` بواسطة حلقة الوكيل قبل المطالبة بالنجاح.
5. الاختبارات التي تغطي: مسار النجاح، والخروج غير الصفري، والمهلة، والثنائي المفقود، واقتطاع الرأس/الذيل الحتمي.

الرفض الصارم:

- `shell=True` في أي مكان في العداء. أرجف فقط.
- الاقتطاع الذي يعتمد على ساعة الحائط أو أخذ العينات العشوائية. يجب أن ينتج نفس الإدخال نفس السجل.
- السجلات بدون `duration_ms`. المجسات البطيئة هي العلامة الأولى لمنضدة العمل الإسفينية.
- محمل يقوم بإرجاع قائمة غير محدودة. ضع حرف C على آخر حرف N أو صفحة.

قواعد الرفض:

- إذا كان المشروع pip يخفي أسرارًا من خلال stdout، ارفض شحن العداء دون خطوة تنقيح. سطح الخطوط التي كان من الممكن أن يتم التقاطها.
- إذا كان المشروع يحتوي على أوامر يمكن تعليقها إلى أجل غير مسمى، فارفض الشحن بدون مهلة افتراضية وقائمة تجاوز صريحة.
- إذا كان العداء يعمل داخل عامل ذي حالة مشتركة، فارفض تخطي قفل الملف حول ملحق JSONL. سيقوم العديد من الكتاب بتمزيق الملف.

هيكل الإخراج:

```
<repo>/
├── feedback_record.jsonl
└── tools/
    ├── run_with_feedback.py
    ├── feedback_loader.py
    └── test_feedback_runner.py
```

انتهي بـ "ما يجب قراءته بعد ذلك" مشيرًا إلى:

- الدرس 38 لبوابة التحقق التي تستهلك السجلات.
- الدرس 39 للوكيل المراجع الذي يقرأ الملاحظات عند تسجيل الجري.
- الدرس 23 لاتفاقيات OTel GenAI لإضافتها إلى جانب القياس عن بعد بمجرد أن تكون التغذية الراجعة قوية.
