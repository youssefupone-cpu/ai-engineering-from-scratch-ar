---
name: skill-mask-rcnn-head-swapper
description: Generate the exact code for swapping box and mask heads on a torchvision Mask R-CNN for a custom num_classes
version: 1.0.0
phase: 4
lesson: 8
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# Mask R-CNN Head Swapper

تنتج لوحة تبديل الرأس للقناع R-CNN على وجه التحديد. يفترض القالب أدناه `model.roi_heads.box_predictor` و `model.roi_heads.mask_predictor`، الموجودين في `maskrcnn_resnet50_fpn` و `maskrcnn_resnet50_fpn_v2` فقط. يحتوي Faster R-CNN على جهاز توقع للصندوق ولكن لا يوجد جهاز توقع للقناع؛ تستخدم RetinaNet `RetinaNetHead` ولا تحتوي على `roi_heads` على الإطلاق — وكلاهما يتطلب مهارات مختلفة.

## When to use

- الضبط الدقيق `maskrcnn_resnet50_fpn` أو `maskrcnn_resnet50_fpn_v2` على مجموعة فئات مخصصة.
- نقل نقطة تفتيش قناع R-CNN تم تدريبها على COCO إلى عدد غير COCO من الفصول.
- تصحيح أخطاء تشغيل قناع R-CNN التدريبي الذي يتعطل عند عدم تطابق `cls_score.out_features` أو `mask_predictor`.

## Out of scope

- `fasterrcnn_*` — لا يوجد قناع_متنبأ. قم بالتبديل `box_predictor` فقط؛ استخدم وصفة منفصلة لتبديل الرأس Faster R-CNN.
- `retinanet_*` — لا `roi_heads`؛ المصنف + رؤوس الانحدار تقع ضمن `model.head.classification_head` و `model.head.regression_head`. استخدم مهارة خاصة بـ RetinaNet.
- `keypointrcnn_*` — يستخدم `keypoint_predictor` بدلاً من `mask_predictor`.

## Inputs

- `model_name`: مُنشئ نموذج كشف torchvision، على سبيل المثال. `maskrcnn_resnet50_fpn_v2`.
- `num_classes`: بما في ذلك الخلفية. مجموعة البيانات المكونة من 4 كائنات تعني `num_classes=5`.
- `freeze`: واحد من `backbone`، `backbone_fpn`، `none`.

## Steps

1. قم باستيراد مُنشئ النموذج وفئتي التوقع (`FastRCNNPredictor`، `MaskRCNNPredictor`).
2. قم بتحميل نموذج الأوزان الافتراضية المُدرب مسبقًا.
3. استبدل `model.roi_heads.box_predictor` بـ `FastRCNNPredictor(in_features, num_classes)` جديد.
4. استبدل `model.roi_heads.mask_predictor` بـ `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)` جديد.
5. تطبيق سياسة التجميد المطلوبة.
6. اطبع كتلة تأكيد تسرد المعلمات القابلة للتدريب لكل وحدة.

## Output code template

```python
from torchvision.models.detection import {MODEL_NAME}, {MODEL_WEIGHTS}
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def build_model(num_classes={NUM_CLASSES}):
    model = {MODEL_NAME}(weights={MODEL_WEIGHTS}.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    {FREEZE_BLOCK}

    return model
```

حيث `{FREEZE_BLOCK}` هو:

- `none` -> فارغة
- `backbone` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- `backbone_fpn` ->
  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  ```

## Report

```
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## Rules

- لا توصي أبدًا بـ `num_classes` بدون تضمين الخلفية؛ تذكير المستخدم دائمًا.
- استخدم دائمًا المتغيرات `_v2` لنماذج الكشف عن torchvision عندما تكون متاحة؛ لديهم أوزان مدربة مسبقًا بشكل أفضل من الأوزان القديمة.
- لا تقم بإنشاء نموذج داخل هذه المهارة - قم بإنتاج كتلة التعليمات البرمجية والسماح للمستخدم بتشغيلها.
- إذا طلب المستخدم `freeze backbone` على مجموعة بيانات أكبر من 10000 صورة، فاقترح عليه أن يفكر في ضبط العمود الفقري أيضًا.
