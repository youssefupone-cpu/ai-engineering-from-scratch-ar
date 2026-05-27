---
name: skill-cmer-monitor
description: Instrument a production VLM endpoint with Cross-Modal Error Rate monitoring, dashboards, and alerts
version: 1.0.0
phase: 4
lesson: 25
tags: [vlm, production, monitoring, hallucination]
---

# CMER Monitor

تعامل مع المحاذاة عبر الوسائط كإنتاج من الدرجة الأولى KPI.

## When to use

- نشر أي نقطة نهاية VLM تنتج نصًا يرتكز على الصور.
- التحقيق في تقارير الاستجابات المهلوسة.
- تتبع ما إذا كان تحول توزيع المدخلات يؤدي إلى تدهور أسس النموذج.

## Inputs

- `vlm_output`: النص الذي تم إنشاؤه.
- `text_confidence`: متوسط ​​الاحتمالية لكل رمز مميز بعد softmax، في `[0, 1]`. احسب كـ `exp(mean(log_probs))`. لا تمر بـ logits الخام؛ logits الخام غير محدود و `conf_threshold` يفترض احتمالًا.
- `image_embedding`: CLIP-التضمين العائلي للصورة (DINOv3, SigLIP, CLIP).
- `text_embedding`: CLIP-التضمين العائلي للنص الذي تم إنشاؤه.
- اختياري `prompt_type`: تسمية للتجميع (vqa / ocr / تسمية توضيحية / وكيل).

## Per-request computation

```python
import torch

def cmer_flag(image_emb, text_emb, text_conf, sim_thr=0.25, conf_thr=0.8):
    if image_emb.shape != text_emb.shape:
        raise ValueError(f"emb shape mismatch: {image_emb.shape} vs {text_emb.shape}")
    image_emb = image_emb / (image_emb.norm() + 1e-8)
    text_emb = text_emb / (text_emb.norm() + 1e-8)
    sim = float((image_emb * text_emb).sum())
    flagged = (text_conf > conf_thr) and (sim < sim_thr)
    return {"sim": sim, "flagged": flagged}
```

التضمينات عبارة عن موترات PyTorch أحادية الأبعاد (`torch.float32`) من جهاز تشفير مستقل من عائلة CLIP. إذا كنت تستخدم المصفوفات NumPy، فاستبدل `.norm()` بـ `np.linalg.norm(...)` وقم بإلقاء الإخراج وفقًا لذلك.

قم بتخزين `sim`، `text_conf`، `flagged`، `prompt_type`، `timestamp`، `model_version`، `request_id` في خط المراقبة الخاص بك pipeline (Prometheus، DataDog، OpenTelemetry).

## Aggregate metric

```
CMER = (flagged requests in window) / (total requests in window)
```

تقرير لكل نقطة نهاية، لكل نوع موجه، لكل إصدار نموذج.

## Alert thresholds

- خط الأساس CMER: إنشاء أكثر من 7 أيام من حركة المرور العادية.
- تحذير: CMER >= 1.5x خط الأساس لمدة ساعة واحدة.
- حاسم: CMER >= 2x خط الأساس لمدة 30 دقيقة أو > 15% مطلق لأي نافذة.

## Dashboard panels

1. CMER بمرور الوقت (مجموعة مدتها 5 دقائق، ونافذة مدتها 7 أيام).
2. CMER بواسطة نوع موجه (شريط مكدس).
3. توزيع `sim` في الساعة (الرسم البياني).
4. أعلى مخرجات الهلوسة (عينة من 20 إجابة محددة يوميًا للمراجعة البشرية).

## Actions when CMER spikes

1. عينة من الطلبات التي تم وضع علامة عليها.
2. تحقق من أن إصدار النموذج لم يتغير عن غير قصد.
3. تحقق من توزيع المدخلات (تنسيق ملف جديد؟ مصدر صورة جديد؟ مضغوط بشكل مختلف؟).
4. قم بتوجيه حركة المرور المتأثرة إلى المراجعة البشرية حتى يتم حل الارتفاع المفاجئ.
5. إذا كان الارتفاع مستمرًا، فقم بضبط النموذج أو استبداله؛ لا تقم بقمع التنبيه.

## Rules

- لا تحسب أبدًا CMER باستخدام التضمينات الخاصة بـ VLM؛ استخدم برنامج تشفير مستقل (DINOv3 أو SigLIP أو CLIP-L/14). وإلا فإنك تقوم بقياس الاتساق الذاتي للنموذج، وليس المحاذاة.
- قم دائمًا بتسجيل القيمة الأولية `sim`، وليس فقط البت `flagged`؛ تظهر تحولات التوزيع في الربع الأدنى قبل أن يتغير معدل العلم.
- لا تشحن نقطة نهاية VLM دون مراقبة CMER؛ الهلوسة هي وضع فشل الإنتاج السائد وهي صامتة بدون هذا المقياس.
- بالنسبة للمجالات الحساسة (الطبية والقانونية والمالية)، ارفع `sim_threshold` إلى 0.35 أو أعلى؛ شرط العلامة هو `sim < sim_threshold`، لذا فإن الحد الأعلى يلتقط المزيد من المخرجات باعتبارها غير مؤرضة - وهو الإعداد الافتراضي الصحيح للاستخدام عالي المخاطر.
