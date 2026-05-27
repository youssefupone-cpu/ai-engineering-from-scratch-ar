---
name: workbench-benchmark
description: Run the same task through prompt-only and workbench-guided pipelines on a project's own sample app and emit a five-outcome before/after report.
version: 1.0.0
phase: 14
lesson: 41
tags: [benchmark, before-after, evaluation, workbench, sample-app]
---

بالنظر إلى الريبو، ومنتج الوكيل، وعينة صغيرة من التطبيق، قم بإنتاج أداة تقييم محمولة تقارن بشكل سريع فقط مع خطوط pipالموجهة إلى طاولة العمل.

Produce:

1. `eval/sample_app/` — نموذج تطبيق قابل للتطبيق بحد أدنى مستمد من مجال المشروع.
2. `eval/run_prompt_only.py` و `eval/run_workbench.py` يأخذ كل منهما وصف المهمة ويعيد `TaskOutcome`.
3. `eval/report.py` يدير كلاً من pipالسطرين ويكتب `before-after-report.md` زائد `comparison.json`.
4. CI سير العمل الذي يفشل عندما تتراجع نتائج طاولة العمل عن مجموعة المهام الثابتة.
5. `docs/benchmark.md` بيان النتائج الخمس وما يعتبر الانحدار.

الرفض الصارم:

- معيار مع خط pipe واحد فقط. المقارنة هي بيت القصيد.
- النتائج مصاغة كنسب مئوية بدون مقام. قم بالإبلاغ دائمًا عن `n / m`.
- نموذج تطبيق تم تدريب منتج الوكيل عليه. استخدم تركيبات مضبوطة على المجال.
- التقارير التي تخفي السلبيات الكاذبة. يجب تعداد المهام التي كانت فيها المطالبة فقط أسرع.

قواعد الرفض:

- إذا لم يكن للمشروع أمر قبول، ارفض شحن المعيار. لا يوجد شيء للقياس.
- إذا كانت طاولة العمل pipeline تستغرق أكثر من 3 أضعاف الخط pipالموجه فقط في المهمة المتوسطة، فقم بإظهار هذه النتيجة؛ طاولة العمل تحتاج إلى التبسيط، وليس النموذج.
- إذا لم يتمكن الحزام من العمل دون الاتصال بالإنترنت، فارفض توصيله بـ CI. سوف يؤدي تقلب الشبكة إلى إفساد المقارنة.

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

- الدرس 42 لحزمة الكابستون التي تجمع كل سطح يستخدمه منضدة العمل pipeline.
- الدرس 19 (SWE-bench، GAIA، AgentBench) لمعايير الماكرو المكملة لهذا.
- الدرس 30 (تطوير الوكيل المبني على التقييم) لحلقات التقييم المستمرة بمجرد ربط المعيار.
