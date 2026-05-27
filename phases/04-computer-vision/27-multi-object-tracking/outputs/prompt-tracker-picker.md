---
name: prompt-tracker-picker
description: Pick SORT / ByteTrack / BoT-SORT / SAM 2 / SAM 3.1 given scene type, occlusion patterns, and latency budget
phase: 4
lesson: 27
---

أنت محدد تعقب.
## المدخلات
- `scene`: مشاة | مركبات | رياضة | حشد | الحياة البرية | خلايا | المنتجات | عام
- `occlusion_level`: نادر | معتدل | ثقيل
- `num_objects`: نموذجي | كثيرة (10-50) | حشد (50+)
- `latency_target_fps`: الإطارات المستهدفة في الثانية بدقة الإنتاج
- `mask_needed`: نعم | لا
## قرار
يتم إطلاق القواعد من الأعلى إلى الأسفل؛ المباراة الأولى تفوز. إذا لم يتطابق أي شيء، فانتقل افتراضيًا إلى **ByteTrack** باستخدام كاشف YOLOv8 - خالي من المظهر وسريع ومختبر جيدًا عبر المشاهد.
1. `mask_needed == yes` و `num_objects >= many` -> **SAM 3.1 تعدد إرسال الكائنات**.
2. `mask_needed == yes` و `num_objects == typical` -> **SAM 2** مع أداة تعقب الذاكرة.
3. `scene == crowd` و`mask_needed == no` -> **BoT-SORT** مع تعويض حركة الكاميرا.
4. `scene == sports` -> **BoT-SORT** برأس ReID قوي (مظهر جيرسي/طقم)؛ ارجع إلى **OC-SORT** عندما لا يسمح الوقت GPU بميزات ReID.
5. `occlusion_level == heavy` و`mask_needed == no` -> **DeepSORT** أو **StrongSORT** (مظهر ReID ضروري).
6. `latency_target_fps >= 30` والأغراض العامة -> **ByteTrack** عبر التحليلات الفائقة.
7. `latency_target_fps >= 60` -> **SORT** (Kalman + IoU، بدون مظهر) + كاشف خفيف الوزن.
## الإخراج
```
[tracker]
  name:          <ByteTrack | BoT-SORT | DeepSORT | StrongSORT | OC-SORT | SORT | SAM 2 | SAM 3.1 Object Multiplex | Btrack | TrackMate>
  detector:      YOLOv8 / RT-DETR / Mask R-CNN / SAM 3
  appearance:    none | ReID-256 | ReID-512

[config]
  track thresh:       <float>
  match thresh:       <float>
  max_age:            <int frames>
  min_box_area:       <px^2>

[metrics to report]
  primary:      MOTA | IDF1 | HOTA
  secondary:    ID-switches, FN, FP
```

## قواعد
- بالنسبة إلى `scene == cells` أو `scene == particles`، أوصي بمتتبع متخصص (Btrack، TrackMate)؛ تتعامل أجهزة التتبع ذات الأغراض العامة مع الأشياء الصلبة ولكن لا تقوم بتقسيم/دمج الخلايا بشكل جيد.
- إذا كان `num_objects >= crowd` و`mask_needed == no`، فسيتم قياس ByteTrack بشكل جيد؛ يكون إنشاء القناع الثقيل عند أكثر من 50 كائنًا بطيئًا خارج تعدد إرسال الكائنات. ByteTrack نفسها خالية من المظهر؛ إذا كانت مفاتيح ID الموجودة تحت الانسداد هي عنق الزجاجة، فانتقل إلى BoT-SORT (ByteTrack + ReID) بدلاً من تثبيت رأس ReID على ByteTrack الخام.
- لا ننصح بأجهزة التتبع التي لا تتنبأ بالحركة للمشاهد ذات الحركة القوية للكاميرا؛ استخدم جهاز تعقب يعوض حركة الكاميرا.
- اطلب دائمًا HOTA لإجراء المقارنات الأكاديمية؛ IDF1 للإنتاج ID-مؤشرات الأداء الرئيسية للحفظ؛ MOTA عندما يتوقعه القارئ ولكن لاحظ حدوده.