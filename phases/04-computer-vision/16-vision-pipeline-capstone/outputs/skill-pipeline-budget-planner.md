---
name: skill-pipeline-budget-planner
description: Given target latency and throughput, assign a time budget to every pipeline stage and flag which stage will miss its budget first
version: 1.0.0
phase: 4
lesson: 16
tags: [vision, pipeline, performance, deployment]
---

# مخطط ميزانية خطوط الأنابيب
قم بتحويل هدف زمن الوصول/الإنتاجية إلى ميزانية خطوة بخطوة حتى يعرف كل عضو في الفريق الرقم الذي يخططون لتحقيقه.
##متى يستخدم
- قبل بناء خدمة الرؤية الجديدة، تحديد التوقعات لكل مرحلة.
- بعد المعيار الأول، لمعرفة أي مرحلة هي الأبعد عن ميزانيتها.
- عندما يلزم إعادة التفاوض بشأن تغييرات SLA والميزانيات.
## المدخلات
- `p95_latency_target_ms`: الميزانية لكل طلب.
- `target_qps`: الإنتاجية لكل نسخة متماثلة.
- `stages`: قائمة `{ name: str, current_ms: float }`.
## قواعد التخصيص
التخصيص الافتراضي عبر المراحل القياسية السبع إذا لم يتم توفير القياسات الحالية:
| المرحلة | شارك |
|-------|-------|
| فك التشفير + المعالجة المسبقة | 15% |
| كاشف للأمام | 55% |
| اكتشافات ما بعد العملية (NMS، المشبك) | 5% |
| اقتصاص + تغيير حجم المصنف | 5% |
| المصنف إلى الأمام | 15% |
| التحقق من صحة المخطط | <1% |
| تسلسل الاستجابة | 4% |
في خطوط pipالمرتبطة بـ GPU (السحابة)، غالبًا ما ترتفع حصة الكاشف إلى 70%. في CPU، تستهلك المعالجة المسبقة وتجميع المصنفات المزيد.
## تقرير
```
[budget plan]
  p95 target:  <ms>
  throughput:  <qps per replica>

| stage               | target_ms | current_ms | headroom | gate |
|---------------------|-----------|------------|----------|------|
| decode+preprocess   | ...       | ...        | ...      | ok|X |
| detector            | ...       | ...        | ...      | ok|X |
| ...                 | ...       | ...        | ...      |      |

[bottleneck]
  stage:  <name>
  miss:   <ms over budget>
  lever:  <specific action>

[levers]
  decode+preprocess:   Pillow-SIMD, libjpeg-turbo, decode on GPU via NVJPEG
  detector:            smaller backbone, lower input resolution, INT8, TensorRT
  postprocess:         GPU-side NMS (torchvision.ops), fused masks
  crop+resize:         GPU crop with grid_sample, batched interpolate
  classifier:          smaller backbone, INT8, warm cache, batch
  schema:              skip validation in hot path, validate at boundaries only
  response:            orjson, stream protobuf
```

## قواعد
- لا تنصح أبدًا بإسقاط التحقق من صحة المخطط من مسار الإنتاج؛ أقترح نقله إلى الحدود بدلا من ذلك.
- إذا فاقت الميزانية المخصصة للمعالجة المسبقة، فجرب دائمًا Pillow-SIMD أو NVJPEG قبل تغيير النموذج.
- إذا كان خطأ الكاشف أكثر من 30% من الهدف، فقم بتبديل النماذج بدلاً من تحسين النموذج الحالي.
- ضع علامة على البوابة كـ `X` عندما يكون current_ms > 1.1 * target_ms; ضع علامة `ok` إذا كانت ضمن 10% من الميزانية.