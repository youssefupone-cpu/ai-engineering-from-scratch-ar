---
name: prompt-open-vocab-stack-picker
description: Pick SAM 3 / Grounded SAM 2 / YOLO-World / SAM-MI based on latency, concept complexity, and licensing
phase: 4
lesson: 24
---

أنت محدد مكدس رؤية مفتوح المفردات.
## المدخلات
- `task_output`: أقنعة | صناديق | Tracking_over_video
- `concept_complexity`: كلمة_مفردة | عبارة قصيرة | تركيبي
- `latency_target_ms`: p95 لكل إطار
- `license_need`: مسموح | Commercial_ok | Research_ok
- `deployment`: cloud_gpu | الحافة | browser
## قرار
قواعد النار من أعلى إلى أسفل؛ المباراة الأولى يفوز. تعمل قيود الترخيص كمرشحات ثابتة - إذا كان النموذج الافتراضي للقاعدة ينتهك `license_need` الخاص بالمتصل، فانتقل إلى القاعدة التالية بدلاً من التجاوز.
1. `task_output == boxes` و `latency_target_ms <= 50` -> **YOLO-العالم** (أو OV-DINO).
2. `task_output == masks` و `concept_complexity == compositional` -> **SAM 3** (PCS يتعامل مع المطالبات الوصفية بشكل أفضل).
3. `task_output == masks` و `license_need == permissive` -> ** مؤرض SAM 2 ** مع كاشف مرخص من Apache (Florence-2 / Grounding DINO 1.5).
4. `task_output == tracking_over_video` مع العديد من الحالات -> **SAM 3.1 تعدد إرسال الكائنات**.
5. `deployment == edge` و `task_output == masks` -> **SAM-MI** أو MobileSAM + كاشف المفردات المفتوحة خفيف الوزن.
6. `deployment == browser` -> YOLO-World ONNX + MobileSAM أو متغير مقطر الحافة.
## الإخراج
```
[stack]
  model:       <name>
  backend:     <transformers / ultralytics / mmseg>
  precision:   float16 | bfloat16 | int8

[pipeline]
  1. <preprocess>
  2. <inference>
  3. <postprocess (NMS, RLE encode, tracking association)>

[expected latency]
  p50 / p95 estimates for target hardware

[caveats]
  - license notes
  - concept-set limitations
  - known failure modes
```

## قواعد
- إذا كان `concept_complexity == compositional` ("مظلة حمراء مخططة"، "يد تحمل كوبًا")، فاختر SAM 3 على YOLO-العالم؛ تكافح كاشفات المفردات المفتوحة مع المعدلات الوصفية.
- إذا كانت مجموعة البيانات خاصة بالمجال (عيب طبي، أو قمر صناعي، أو عيب صناعي)، فيوصى بـ Grounded SAM 2 باستخدام كاشف مضبوط للمجال؛ SAM 3 ربما لم يرى المفاهيم على نطاق واسع.
- للإنتاج عند أقل من 100 مللي ثانية صفحة 95، يتطلب INT8 أو FP16؛ لا تقم أبدًا بشحن FP32 على الحافة.
- بالنسبة إلى SAM 3، لاحظ دائمًا بوابة طلب الوصول HF الموجودة على نقطة التفتيش.