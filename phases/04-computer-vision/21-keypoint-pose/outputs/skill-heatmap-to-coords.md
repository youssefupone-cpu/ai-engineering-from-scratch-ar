---
name: skill-heatmap-to-coords
description: Write the sub-pixel heatmap-to-coordinate routine used by every production pose model
version: 1.0.0
phase: 4
lesson: 21
tags: [keypoint, pose, subpixel, inference]
---

# Heatmap to Coords

قم بتحويل خرائط الحرارة الأولية للنقاط الرئيسية إلى إحداثيات دقيقة لوحدات البكسل الفرعية. أرخص ترقية للدقة في كل وضعية pipeline.

## When to use

- نشر نموذج النقاط الرئيسية القائم على الخريطة الحرارية.
- قياس مقاييس الوضع — OKS حساس للغاية لدقة البكسل الفرعي.
- نقل رمز الوضع من إطار إلى آخر.

## Inputs

- `heatmaps`: `(N, K, H, W)` خرائط حرارية لكل نقطة مفتاح من النموذج.
- `confidence_threshold`: تجاهل النقاط الرئيسية التي تكون ذروتها أقل من هذه القيمة.

## Steps

1. **Argmax** كل خريطة حرارية للعثور على موقع الذروة الصحيح.
2. **إزاحة الفرق الأول** — تقدير إزاحة البكسل الفرعي من وحدات البكسل المجاورة. المعامل `0.25` عبارة عن إرشادي تمت معايرته لخرائط الحرارة الغوسية باستخدام `sigma >= 1`؛ لاستعادة وحدات البكسل الفرعية المبدئية، استخدم الملاءمة التربيعية الكاملة (DARK) أو الملاءمة الغوسية.

```
dx = 0.25 * sign(heatmap[y, x+1] - heatmap[y, x-1])
dy = 0.25 * sign(heatmap[y+1, x] - heatmap[y-1, x])
```

بالنسبة للمتغير التربيعي DARK /، يمكنك التقريب باستخدام المعادلة التربيعية المحلية:

```
dx = -0.5 * (heatmap[y, x+1] - heatmap[y, x-1])
        / (heatmap[y, x+1] - 2 * heatmap[y, x] + heatmap[y, x-1] + eps)
```

يكون التوافق التربيعي أكثر دقة في الخرائط الحرارية ذات الذروة؛ تعتبر الإزاحة المستندة إلى الإشارة هي الخيار الافتراضي الأكثر أمانًا عندما تكون الخرائط الحرارية مزعجة.

3. **أضف الإزاحة** إلى ذروة العدد الصحيح.
4. **الثقة** — إرجاع قيمة الذروة لكل نقطة مفاتيح؛ يستخدمه العملاء لإخفاء توقعات الثقة المنخفضة.
5. **حالة الحدود** — عندما تهبط الذروة على أول أو آخر بكسل على طول المحور، يتم تثبيت أحد الجيران؛ ينهار الإزاحة إلى الصفر، وهو الإجراء الاحتياطي الأكثر أمانًا.

## Output template

```python
import torch

def heatmap_to_coords_subpixel(heatmaps, threshold=0.2):
    N, K, H, W = heatmaps.shape
    flat = heatmaps.reshape(N, K, -1)
    conf, idx = flat.max(dim=-1)
    ys = (idx // W).float()
    xs = (idx % W).float()

    ys_int = ys.long()
    xs_int = xs.long()

    x_minus = (xs_int - 1).clamp(min=0)
    x_plus = (xs_int + 1).clamp(max=W - 1)
    y_minus = (ys_int - 1).clamp(min=0)
    y_plus = (ys_int + 1).clamp(max=H - 1)

    batch_idx = torch.arange(N).view(-1, 1).expand(-1, K)
    kp_idx = torch.arange(K).view(1, -1).expand(N, -1)

    dx_raw = (heatmaps[batch_idx, kp_idx, ys_int, x_plus]
              - heatmaps[batch_idx, kp_idx, ys_int, x_minus])
    dy_raw = (heatmaps[batch_idx, kp_idx, y_plus, xs_int]
              - heatmaps[batch_idx, kp_idx, y_minus, xs_int])
    dx = 0.25 * torch.sign(dx_raw)
    dy = 0.25 * torch.sign(dy_raw)

    at_left = xs_int == 0
    at_right = xs_int == (W - 1)
    at_top = ys_int == 0
    at_bottom = ys_int == (H - 1)
    dx = torch.where(at_left | at_right, torch.zeros_like(dx), dx)
    dy = torch.where(at_top | at_bottom, torch.zeros_like(dy), dy)

    refined_x = xs + dx
    refined_y = ys + dy
    coords = torch.stack([refined_x, refined_y], dim=-1)
    mask = conf >= threshold
    return coords, conf, mask
```

## Report

```
[subpixel decode]
  keypoints:   K
  threshold:   <float>
  valid_rate:  fraction of keypoints above threshold
```

## Rules

- قم دائمًا بربط مؤشرات الجوار بنطاق صالح؛ نقاط المفاتيح البعيدة عن الحافة لها إزاحة فرق صفري ولكن لا يوجد بها أي عطل.
- إعادة الثقة إلى جانب الإحداثيات حتى يتمكن العملاء من إخفاء نقاط الثقة المنخفضة.
- يساعد تحسين البكسل الفرعي فقط عندما تكون الخريطة الحرارية سلسة حول الذروة - تأكد من أن التدريب استخدم هدفًا غاوسيًا مع سيجما >= 1.
- بالنسبة إلى دقة الخريطة الحرارية الصغيرة جدًا (< 48x48)، فكر في تكبير حجم الخريطة الحرارية إلى الحجم الكامل للصورة قبل استخراج الإحداثيات؛ يتم قياس إزاحة البكسل الفرعي مع الخطوة.
