---
name: skill-depth-to-pointcloud
description: Build point clouds from depth maps with correct intrinsics handling and export to .ply
version: 1.0.0
phase: 4
lesson: 26
tags: [depth, point-cloud, 3d, intrinsics]
---

# Depth to Point Cloud

قم بتحويل خريطة العمق بالإضافة إلى صورة ملونة إلى سحابة نقطية مزخرفة، قابلة للتصدير للتصور أو المزيد من العمل ثلاثي الأبعاد.

## When to use

- تصور تنبؤات العمق كمشهد ثلاثي الأبعاد فعلي.
- إعادة بناء متفرقة ثلاثية الأبعاد من صورة واحدة.
- إنتاج مدخلات لتدريب 3DGS عند فشل SfM.
- مقارنة العمق المتوقع مع الحقيقة الأرضية بتقنية LiDAR.

## Inputs

- `depth`: `(H, W)` مجموعة من الأعماق في نفس الوحدات التي تريدها في الإخراج (يوصى بالأمتار).
- `rgb`: `(H, W, 3)` مجموعة من الألوان (uint8 أو float32 [0, 1]).
- `intrinsics`: `(fx, fy, cx, cy)` بوحدات البكسل.
- اختياري `depth_scale`: مضاعف لتحويل وحدات العمق المتوقعة إلى أمتار.

## Pipeline

1. **التحقق** — يجب أن يكون العمق إيجابيًا ومحدودًا في كل مكان تخطط لتضمينه. إخفاء وحدات البكسل غير الصالحة.
2. **الرفع** — `X = (u - cx) * d / fx`، `Y = (v - cy) * d / fy`، `Z = d` لكل بكسل.
3. **إقران** مع RGB — تحصل كل نقطة ثلاثية الأبعاد على `(r, g, b)` ثلاثية من البكسل المطابق.
4. **تصدير** — PLY (محمول)، `.xyz` (خفيف الوزن)، `.pcd` (Open3D أصلي)، `.las`/`.laz` (جغرافي مكاني).

## Implementation template

```python
import numpy as np

def depth_to_point_cloud(depth, intrinsics, depth_scale=1.0, min_depth=0.1, max_depth=100.0):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    z = depth.astype(np.float32) * depth_scale
    valid = (z > min_depth) & (z < max_depth) & np.isfinite(z)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    return points, valid


def write_ply(path, points, colors=None, valid_mask=None):
    p = points.reshape(-1, 3)
    if valid_mask is not None:
        p = p[valid_mask.flatten()]
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {p.shape[0]}",
        "property float x", "property float y", "property float z",
    ]
    if colors is not None:
        c = colors.reshape(-1, 3).astype(np.uint8)
        if valid_mask is not None:
            c = c[valid_mask.flatten()]
        lines += ["property uchar red", "property uchar green", "property uchar blue"]
    lines.append("end_header")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        if colors is not None:
            for pt, col in zip(p, c):
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f} {col[0]} {col[1]} {col[2]}\n")
        else:
            for pt in p:
                f.write(f"{pt[0]:.4f} {pt[1]:.4f} {pt[2]:.4f}\n")
```

## Report

```
[export]
  input depth shape:  (H, W)
  valid points:       <N> of <H*W>
  output format:      ply | xyz | pcd | las
  coordinate system:  camera (+X right, +Y down, +Z forward)
  scale:              metres | millimetres | normalised
```

## Rules

- قم دائمًا بإخفاء العمق غير الصالح (صفر، NaN، inf، مشبع)؛ بما في ذلك ينتج سحابة من نقاط القمامة في الأصل.
- للتنبؤ من نموذج العمق النسبي، قم بالتصدير NOT كمقياس؛ اسم ملف الإخراج البادئة مع `relative_` للإشارة إلى الاتفاقية.
- حافظ على اتساق اصطلاح تنسيق الكاميرا (OpenCV: +X لليمين، +Y للأسفل، +Z للأمام). قم بتبديل الإشارات إذا كانت الأداة المتلقية للمعلومات تتوقع OpenGL (+Y لأعلى).
- بالنسبة للمشاهد الكثيفة (> 1 مليون نقطة)، قم بتقديم معلمة عينة فرعية؛ PLY الملفات > 500 MB يصعب تحميلها في كل مكان.
- لا تقم مطلقًا بقص العمق بصمت لإنتاج مخرجات "معقولة"؛ قص بشكل صريح مع الحدود التحذيرية حتى يعرف المستخدمون ما تم التخلص منه.
