---
name: skill-freeze-inspector
description: Report which parameters are trainable, which BatchNorm layers are in eval mode, and whether the optimizer is actually consuming the trainable parameters
version: 1.0.0
phase: 4
lesson: 5
tags: [computer-vision, transfer-learning, debugging, pytorch]
---

# تجميد المفتش
تختبئ أخطاء التعلم بالنقل في ثلاثة أماكن: المعلمات التي يجب تجميدها ولكنها ليست كذلك، والمعلمات التي يجب أن تكون قابلة للتدريب ولكنها ليست كذلك، والمحسنات التي تم إنشاؤها قبل تغيير حالة التجميد. تظهر هذه المهارة الثلاثة في تمريرة واحدة.
##متى يستخدم
- مباشرة بعد ضبط `requires_grad` على مجموعة فرعية من المعلمات.
- قبل الخطوة التدريبية الأولى للجري الدقيق.
- بعد الاتصال بـ `freeze_bn_stats` أو أي مساعد يقوم بقلب وضع BN.
- عندما تكون دقة val عالقة بشكل عشوائي وتشك في عدم وجود أي تدريب فعليًا.
## المدخلات
- `model`: PyTorch `nn.Module`.
- `optimizer`: المحسن الذي سيتم استخدامه للتدريب.
- اختياري `expected_frozen_prefixes`: قائمة ببادئات أسماء المعلمات التي يجب تجميدها (على سبيل المثال `["conv1", "bn1", "layer1"]`).
## الخطوات
1. **معلمات السير.** لكل `(name, param)`:
   - سجل `requires_grad`
   - سجل `shape` و`numel`
2. **وحدات المشي.** لكل وحدة:
   - إذا كان BatchNorm، سجل ما إذا كان في وضع التقييم وما إذا كانت معلماته المتقاربة قابلة للتدريب.
3. **فحص المُحسِّن.** لكل مجموعة معلمات:
   - قم بتسوية `params` في مجموعة من `id(p)`.
   - قارن مع مجموعة كل `id(p)` للمعلمات حيث `requires_grad == True`.
4. **اكتشف أوضاع الفشل الأربعة:**
   - `leaked_train`: تحتوي المعلمة على `requires_grad=True` ولكنها لا تظهر في المُحسِّن (يتم حساب التدرج ولكن لا يتم تطبيقه مطلقًا).
   - `ghost_train`: تظهر معلمة في المُحسِّن ولكنها تحتوي على `requires_grad=False` (يتم إهدار حالة المُحسِّن؛ ويمكن أن تتسبب أيضًا في حدوث أخطاء إذا قمت بإعادة التمكين لاحقًا يتطلب_grad).
   - `bn_mismatch`: إما (أ) أن تكون طبقة BN في وضع التدريب (تتراكم إحصائيات التشغيل) بينما تكون المعلمات التابعة لها (`weight`، `bias`) مجمدة، أو (ب) تكون الطبقة BN في وضع التقييم (الإحصائيات المجمدة) بينما تكون المعلمات التابعة لها قابلة للتدريب. كلتا الحالتين غير متناسقتين ودائمًا ما يكون هناك خطأ.
   - `expected_vs_actual`: أي بادئة مدرجة في `expected_frozen_prefixes` لا تزال تحتوي على معلمة قابلة للتدريب.
## تقرير
```
[freeze-inspector]
  model trainable params: <N>
  model frozen params:    <N>
  batchnorm layers in eval mode: <count>
  batchnorm layers in train mode: <count>

[optimizer coverage]
  trainable params fed to optimizer: <M> of <N>
  leaked_train: <list of names> (trainable but not in optimizer)
  ghost_train:  <list of names> (in optimizer but frozen)

[bn audit]
  mismatched layers: <list of names>

[expectations]
  expected_frozen_prefixes: <...>
  violating params:         <list>

[verdict]
  ok | <one-line summary of the most severe issue>
```

## قواعد
- أسماء معلمات التقرير فقط؛ لا تطبع الأوزان نفسها أبدًا.
- فرز كل قائمة أبجديا حسب اسم المعلمة.
- إذا كانت تغطية المحسن 100% ولا يوجد أي عدم تطابق، قم بإرجاع `ok` وتوقف.
- بالنسبة إلى `leaked_train`، ننصح دائمًا بإعادة بناء المُحسِّن بعد تغيير حالة التجميد.
- بالنسبة إلى `ghost_train`، نوصي بإزالة مجموعة المعلمات أو الإعداد `requires_grad=True` إذا كان القصد هو تدريبها.