---
name: skill-mask-rcnn-head-swapper
description: Generate the exact code for swapping box and mask heads on a torchvision Mask R-CNN for a custom num_classes
version: 1.0.0
phase: 4
lesson: 8
tags: [computer-vision, mask-rcnn, fine-tuning, torchvision]
---

# قناع R-CNN مقايضة الرأس
يتم إنتاج نموذج تبديل الرأس للقناع R-CNN على وجه التحديد. يفترض القالب أدناه `model.roi_heads.box_predictor` و`model.roi_heads.mask_predictor`، الموجودين في `maskrcnn_resnet50_fpn` و`maskrcnn_resnet50_fpn_v2` فقط. يحتوي R-CNN الأسرع على توقع للمربع ولكن لا يوجد توقع للقناع؛ تستخدم RetinaNet `RetinaNetHead` ولا تحتوي على `roi_heads` على الإطلاق - وكلاهما يتطلب مهارات مختلفة.
##متى يستخدم
- الضبط الدقيق `maskrcnn_resnet50_fpn` أو `maskrcnn_resnet50_fpn_v2` على مجموعة فئات مخصصة.
- نقل نقطة تفتيش القناع R-CNN المدربة على COCO إلى عدد فصول غير COCO.
- تصحيح أخطاء تشغيل تدريب القناع R-CNN الذي يتعطل عند عدم تطابق `cls_score.out_features` أو `mask_predictor`.
## خارج النطاق
- `fasterrcnn_*` — لا يوجد قناع_متنبأ. قم بالتبديل فقط `box_predictor`; استخدم وصفة أسرع R-CNN لتبديل الرأس.
- `retinanet_*` — لا `roi_heads`؛ المصنف + رؤوس الانحدار موجودة تحت `model.head.classification_head` و `model.head.regression_head`. استخدم مهارة خاصة بـ RetinaNet.
- `keypointrcnn_*` — يستخدم `keypoint_predictor` بدلاً من `mask_predictor`.
## المدخلات
- `model_name`: مُنشئ نموذج كشف torchvision، على سبيل المثال. __الكود_1__.
- `num_classes`: بما في ذلك الخلفية. مجموعة البيانات المكونة من 4 كائنات تعني `num_classes=5`.
- `freeze`: واحد من `backbone`، `backbone_fpn`، `none`.
## الخطوات
1. قم باستيراد منشئ النموذج وفئتي التوقع (`FastRCNNPredictor`، `MaskRCNNPredictor`).
2. قم بتحميل نموذج الأوزان الافتراضية المُدرب مسبقًا.
3. استبدل `model.roi_heads.box_predictor` بـ `FastRCNNPredictor(in_features, num_classes)` الجديد.
4. استبدل `model.roi_heads.mask_predictor` بـ `MaskRCNNPredictor(in_features_mask, hidden_layer=256, num_classes)` الجديد.
5. تطبيق سياسة التجميد المطلوبة.
6. اطبع كتلة تأكيد تسرد المعلمات القابلة للتدريب لكل وحدة.
## قالب كود الإخراج
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
- `none` -> فارغ
- __الكود_1__ ->  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  ```
- __الكود_0__ ->  ```python
  for p in model.backbone.parameters():
      p.requires_grad = False
  # FPN parameters live inside backbone.fpn
  ```

## تقرير
```
[head-swap]
  model:         <MODEL_NAME>
  num_classes:   <N>  (includes background)
  freeze policy: <choice>
  trainable:     <N>
  total:         <N>
```

## قواعد
- لا توصي أبدًا بـ `num_classes` بدون الخلفية المضمنة؛ تذكير المستخدم دائمًا.
- استخدم دائمًا متغيرات `_v2` لنماذج الكشف عن torchvision عندما تكون متاحة؛ لديهم أوزان مدربة مسبقًا بشكل أفضل من الأوزان القديمة.
- لا تقم بإنشاء نموذج داخل هذه المهارة - قم بإنتاج كتلة التعليمات البرمجية والسماح للمستخدم بتشغيلها.
- إذا طلب المستخدم `freeze backbone` على مجموعة بيانات أكبر من 10000 صورة، فاقترح عليه أن يفكر في ضبط العمود الفقري أيضًا.