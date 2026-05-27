---
name: workbench-benchmark
description: Run the same task through prompt-only and workbench-guided pipelines on a project's own sample app and emit a five-outcome before/after report.
version: 1.0.0
phase: 14
lesson: 41
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

بالنظر إلى الريبو، ومنتج الوكيل، وعينة تطبيق صغيرة، قم بإنتاج أداة تقييم محمولة تقارن بشكل فوري فقط مع خطوط pipe الموجهة إلى طاولة العمل.
ينتج:
1. `eval/sample_app/` — نموذج تطبيق قابل للتطبيق بحد أدنى مستمد من مجال المشروع.
2. `eval/run_prompt_only.py` و`eval/run_workbench.py` يأخذ كل منهما وصفًا للمهمة ويعيد `TaskOutcome`.
3. `eval/report.py` الذي يقوم بتشغيل كل من سطر pip ويكتب `before-after-report.md` بالإضافة إلى `comparison.json`.
4. CI سير العمل الذي يفشل عندما تتراجع نتائج طاولة العمل في مجموعة المهام الثابتة.
5. `docs/benchmark.md` شرح النتائج الخمس وما يعتبر انحدارا.
الرفض الصارم:
- معيار مرجعي يحتوي على سطر pipe واحد فقط. المقارنة هي بيت القصيد.
- النتائج مصاغة كنسب مئوية بدون مقام. أبلغ دائمًا عن `n / m`.
- نموذج تطبيق تم تدريب منتج الوكيل عليه. استخدم تركيبات مضبوطة على المجال.
- التقارير التي تخفي السلبيات الكاذبة. يجب تعداد المهام التي كانت فيها المطالبة فقط أسرع.
قواعد الرفض:
- إذا لم يكن للمشروع أمر قبول، ارفض شحن المعيار. لا يوجد شيء للقياس.
- إذا كان منضدة العمل pipeline تستغرق أكثر من 3x من خط pipeline للمطالبة فقط في المهمة المتوسطة، فاعرض هذه النتيجة؛ طاولة العمل تحتاج إلى التبسيط، وليس النموذج.
- إذا لم يكن من الممكن تشغيل مجموعة الأسلاك دون الاتصال بالإنترنت، فارفض توصيلها بـ CI. سوف يؤدي تقلب الشبكة إلى إفساد المقارنة.
هيكل الإخراج:
```
<repo>/
├── eval/
│   ├── sample_app/
│   ├── run_prompt_only.py
│   ├── run_workbench.py
│   └── report.py
├── outputs/eval/
│   ├── before-after-report.md
│   └── comparison.json
├── docs/benchmark.md
└── .github/workflows/benchmark.yml
```

انتهي بـ "ما يجب قراءته بعد ذلك" مشيرًا إلى:
- الدرس 42 لحزمة الأغطية التي تجمع كل سطح يستخدمه طاولة العمل pipeline.
- الدرس 19 (SWE-bench، GAIA، AgentBench) لمقاييس الماكرو المكملة لهذا.
- الدرس 30 (تطوير الوكيل المبني على التقييم) لحلقات التقييم المستمرة بمجرد ربط المعيار.