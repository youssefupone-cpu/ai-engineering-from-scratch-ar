---
name: skill-frame-sampler-auditor
description: Audit a video pipeline's frame sampler for off-by-one, short-clip handling, and crop consistency
version: 1.0.0
phase: 4
lesson: 12
tags: [computer-vision, video, sampling, debugging]
---

# Frame Sampler Auditor

أخذ عينات الإطار هو المكان الذي تنقطع فيه خطوط الفيديو pip. تنتشر الأخطاء هنا في كل مقياس نهائي.

## When to use

- كتابة محمل بيانات فيديو جديد.
- إعادة إنتاج الأرقام من ورقة ودقة التدريب أقل من المبلغ عنها.
- تصحيح أخطاء نموذج فيديو تكون دقة تقييمه غير مستقرة عبر عمليات التشغيل.

## Inputs

- `sampler_code`: دالة بايثون تأخذ (num_frames_total, T) وترجع مؤشرات T.
- `T`: طول المقطع المستهدف.
- حالات اختبار اختيارية: `num_frames_total` القيم المراد تمرينها (على سبيل المثال `[3, T-1, T, T+1, 30, 300, 3000]`).

## Checks

### 1. Short clip handling
Feed `num_frames_total < T`. Every returned index must be in `[0, num_frames_total - 1]`. The standard padding policy is to repeat the last frame for the remaining positions.

### 2. Boundary indices
Feed `num_frames_total == T`. Returned indices should be `[0, 1, ..., T-1]` exactly.

### 3. Uniform distribution
Feed `num_frames_total == 10 * T`. Returned indices should be monotonically increasing and roughly evenly spaced.

### 4. Dense window bounds
For dense sampling, feed `num_frames_total == 3 * T`. Returned indices should form a contiguous window, never crossing the end of the clip.

### 5. Determinism
Call the sampler twice with the same inputs and (for deterministic samplers) the same RNG. Indices should match.

### 6. Crop consistency
If the pipeline also returns a spatial crop per frame, run the sampler twice for the same clip with the same seed and confirm every frame uses the same crop box (same `(x, y, w, h)`). Different crops per frame inside one clip destroys temporal coherence and is a classic silent bug. Acceptable variation: augmentation applied *per clip*, consistent within a clip.

## Report

```
[sampler audit]
  name: <function name>
  T:    <int>

[short-clip handling]
  passed | failed (<details>)

[boundary]
  passed | failed

[uniform spacing]
  passed | failed (<stddev of gaps>)

[dense window]
  passed | failed (<details>)

[determinism]
  passed | failed

[crop consistency]
  passed | failed (<per-frame crop varies: yes/no>)

[verdict]
  ok | fix required
```

## Rules

- لا تضع أبدًا علامة "موافق" على جهاز أخذ العينات إذا كانت معالجة المقطع القصير تؤدي إلى إرجاع مؤشرات خارج النطاق.
- يجب ألا تقوم أجهزة أخذ العينات الكثيفة أبدًا بإرجاع نافذة تتقاطع مع `num_frames_total - 1`.
- إذا كانت العينة عشوائية (كثيفة)، فاختبر الحتمية فقط باستخدام بذرة صريحة RNG.
- اقترح، ولكن لا تقم بإصلاح السياسات الأساسية بصمت: لوحة مع الإطار الأخير، نافذة مثبتة حتى النهاية، فواصل زمنية نصف مفتوحة مستديرة.
