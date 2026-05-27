---
name: skill-physical-plausibility-checks
description: Automated checks for object permanence, gravity, and continuity on any generated video before shipping
version: 1.0.0
phase: 4
lesson: 28
tags: [video-generation, quality, physics, evaluation]
---

# Physical Plausibility Checks

تحتاج عمليات نشر إنتاج الفيديو الذي تم إنشاؤه إلى حواجز حماية آلية. المراجعة البشرية لا تتسع؛ تكتشف الفحوصات الفيزيائية أوضاع الفشل الكلاسيكية.

## When to use

- أي منتج يقوم بإنشاء فيديو من مطالبات نصية أو صورية.
- أتمتة QA على نقطة نهاية إنشاء الفيديو API.
- مراقبة انحراف جودة نموذج الفيديو بعد الضبط الدقيق أو تحديث النموذج الأساسي.

## Inputs

- `video`: موتر `(T, H, W, 3)` أو مسار إلى mp4.
- معلومات مرجعية اختيارية: العدد المتوقع من الكائنات، ووصف المشهد الأولي.

## Checks

### 1. Object permanence
Track every detection across frames with SAM 3.1 Object Multiplex. Flag when a stable track disappears for <=3 frames and reappears — the model lost the object temporarily. Hard fail when an object disappears near the frame centre (not at an edge); soft fail at edges.

### 2. Motion smoothness
Optical flow between consecutive frames should be mostly continuous. Sudden per-pixel flow spikes indicate teleportation. Compute flow with RAFT; flag frames where the 99th-percentile flow magnitude exceeds the median by a factor > 10.

### 3. Gravity / support
For objects detected as solid (food, balls, tools), check that their vertical position is non-increasing in the absence of a lifting action. Flag upward drift unless a "grasping hand" is detected near the object.

### 4. Identity consistency
For people or characters, use a face-recognition embedding across frames. Cosine similarity should stay > 0.8 across 5-frame windows for a persistent identity. Below threshold means the character morphed.

### 5. Hands and limbs
Run a pose estimator (Lesson 21). Flag frames where a hand has > 5 or < 4 visible fingers; where an arm length doubles between frames; where limbs intersect the body through a surface.

### 6. Text rendering (if prompt asked for text)
If the user prompt included a string in quotes, OCR the generated frames and compute CER against the requested string. Flag > 20% CER.

## Report

```
[plausibility]
  video frames:           <T>
  permanence violations:  <N>
  smoothness violations:  <N>
  gravity violations:     <N>
  identity drift:         <N of 5-frame windows>
  limb anomalies:         <N>
  OCR CER vs requested:   <float>

[verdict]
  ship | hold | reject

[samples for review]
  frame ranges where each failure occurred
```

## Rules

- لا تحظر بشدة على أي شيك واحد؛ قم بتجميع الدرجات واحتفظ بالفيديو للمراجعة عندما يتجاوز إجمالي الحالات الشاذة الحد الأدنى.
- إنحراف هوية الوزن وانتهاكات الدوام هي الأعلى — يلاحظها المستخدمون أولاً.
- تسجيل معدلات فشل كل فحص مع مرور الوقت؛ يعني الاتجاه الصاعد عادةً أنه تم تحديث النموذج الأساسي أو تحول التوزيع الفوري.
- لا تقم أبدًا بحذف الفيديو الذي تم الإبلاغ عنه؛ احتفظ به لتصحيح أخطاء النماذج وعمليات التشريح بعد الوفاة.
- بالنسبة للمحتوى الحساس (الأشخاص، الأطفال، الشخصيات العامة)، يتطلب مراجعة بشرية لكل فيديو بغض النظر عن النتيجة.
