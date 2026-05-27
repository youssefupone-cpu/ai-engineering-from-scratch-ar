---
name: skill-mot-evaluator
description: Write a complete evaluation harness for MOTA / IDF1 / HOTA against ground-truth tracks
version: 1.0.0
phase: 4
lesson: 27
tags: [mot, evaluation, tracking, metrics]
---

# MOT Evaluator

قم بلف مخرجات جهاز التعقب الخاص بك في الخط القياسي MOTA/IDF1/HOTA pipeline حتى تتمكن من المقارنة بشكل عادل مع الأدبيات.

## When to use

- قياس أداء المتتبع الجديد على MOT17 / MOT20 / DanceTrack / SportsMOT.
- مقارنة ByteTrack بـ BoT-SORT إلى SAM 2 على اللقطات الخاصة بك.
- إنتاج رقم قابل للتكرار للورقة أو وصف PR.

## Inputs

- `predictions`: قائمة لكل إطار من الصفوف `(track_id, x, y, w, h, confidence)`.
- `ground_truth`: قائمة لكل إطار من `(gt_id, x, y, w, h)` صفوف.
- `iou_threshold`: 0.5 نموذجي لـ MOTA؛ HOTA يستخدم عملية المسح.
- `evaluator`: `py-motmetrics` (MOTA، IDF1) أو `TrackEval` (HOTA).

## Output format contract

يتوقع كل من `py-motmetrics` و `TrackEval` تنسيقًا محددًا على القرص:

```
# predictions.txt
<frame>,<track_id>,<x>,<y>,<w>,<h>,<confidence>,-1,-1,-1

# ground_truth.txt
<frame>,<gt_id>,<x>,<y>,<w>,<h>,1,-1,-1,-1
```

الإطارات مفهرسة بـ 1، والمربعات هي (x، y، w، h)، وليس (x1، y1، x2، y2). التحويل هو المكان الذي تعيش فيه معظم أخطاء التكامل.

## Steps

1. قم بتحويل مخرجات جهاز التعقب الخاص بك إلى تنسيق نص التحدي MOT.
2. قم بتشغيل `py-motmetrics.io.loadtxt` على كلا الملفين.
3. احسب MOTA + IDF1 باستخدام `mm.metrics.create().compute()`.
4. بالنسبة إلى HOTA، قم باستدعاء `TrackEval` بنفس الملفات و`Metrics: HOTA`.
5. حفظ النتائج باسم JSON للوحات المعلومات.

## Implementation sketch

```python
import motmetrics as mm

def evaluate_mota_idf1(pred_path, gt_path):
    gt = mm.io.loadtxt(gt_path, fmt="mot15-2D")
    pred = mm.io.loadtxt(pred_path, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, pred, dist="iou", distth=0.5)
    metrics = mm.metrics.create().compute(
        acc, metrics=["num_frames", "mota", "motp", "idf1", "idp", "idr", "num_switches"]
    )
    return metrics


def write_mot_txt(predictions, path):
    with open(path, "w") as f:
        for frame_idx, detections in enumerate(predictions, start=1):
            for tid, x, y, w, h, conf in detections:
                f.write(f"{frame_idx},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.3f},-1,-1,-1\n")
```

## Report

```
[mot evaluation]
  frames:     <int>
  gt tracks:  <int>
  pred tracks: <int>

[metrics]
  MOTA:       <float>
  MOTP:       <float>
  IDF1:       <float>
  IDP/IDR:    <float/float>
  ID switches: <int>
  HOTA:       <float>  (from TrackEval)
```

## Rules

- استخدم دائمًا الإطارات المفهرسة 1 في الملف النصي الناتج؛ MOT الأدوات تتوقع ذلك.
- تحويل (x1، y1، x2، y2) إلى (x، y، w، h) قبل الكتابة.
- لا تبلغ MOTA وحدك للمقارنات الحديثة؛ تشمل IDF1 و HOTA.
- راقب الاكتشافات الخاصة والعامة على MOT17 - يتم تقييمها بشكل منفصل ويؤدي خلطها إلى تضخيم النتائج.
- سجل العشرات في التسلسل. يخفي التجميع حالات الفشل في تسلسلات فردية صعبة.
