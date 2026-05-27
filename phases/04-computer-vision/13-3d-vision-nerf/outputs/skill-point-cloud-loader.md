---
name: skill-point-cloud-loader
description: Write a PyTorch Dataset for .ply / .pcd / .xyz files with correct normalisation, centring, and point sampling
version: 1.0.0
phase: 4
lesson: 13
tags: [3d-vision, point-cloud, data-loading, pytorch]
---

# نقطة محمل السحابة
قم بتحويل مجلد من ملفات المسح الضوئي ثلاثي الأبعاد إلى مجلد جاهز للتدريب PyTorch `Dataset`.
##متى يستخدم
- بدء مشروع جديد لتصنيف/تجزئة السحابة النقطية.
- التبديل بين صيغ `.ply`، `.pcd`، و`.xyz`.
- تصحيح أخطاء النموذج الذي يتدرب بدون أخطاء ولكنه يتقارب بشكل سيئ؛ غالبًا ما يكون تطبيع أداة تحميل البيانات خاطئًا.
## المدخلات
- `data_root`: مجلد ملفات Point-Cloud وCSV اختياري مع التسميات.
- `file_format`: رقائق | بي سي دي | سز | npy.
- `num_points`: حجم العينة الثابت، عادةً 1024 أو 2048.
- `augmentation`: لا يوجد | تدوير | غضب | mixup.
## سياسة التطبيع
تنطبق كل سحابة نقطة إنتاج pipeline بالترتيب:
1. **مركز** السحابة: اطرح النقطه الوسطى.
2. **المقياس** إلى وحدة الكرة: قسّم على أقصى مسافة من المركز.
3. **نموذج** `num_points` نقطة. إذا كانت السحابة تحتوي على المزيد، فاستخدم **أخذ عينات من النقطة الأبعد** (FPS) لتمثيل الشكل بدقة أو أخذ عينات عشوائية للسرعة. إذا كان أقل، كرر النقاط.
4. **الخلط العشوائي** ترتيب النقاط (لا ينبغي أن يكون الترتيب مهمًا بالنسبة للنموذج على أية حال، ولكن الترتيب العشوائي يكسر تبعيات الترتيب غير المقصودة).
## قالب الإخراج
```python
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

def _read_ply(path):
    if HAS_O3D:
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    # Fallback: minimal ascii-ply reader
    ...

def _fps(points, k):
    idx = np.zeros(k, dtype=np.int64)
    dist = np.full(len(points), np.inf)
    seed = np.random.randint(len(points))
    idx[0] = seed
    for i in range(1, k):
        dist = np.minimum(dist, ((points - points[idx[i-1]]) ** 2).sum(axis=1))
        idx[i] = int(np.argmax(dist))
    return idx

def normalise(points):
    centre = points.mean(axis=0)
    points = points - centre
    scale = np.max(np.linalg.norm(points, axis=1))
    return points / max(scale, 1e-8)

class PointCloudDataset(Dataset):
    def __init__(self, files, labels, num_points=1024, augment=False):
        self.files = files
        self.labels = labels
        self.num_points = num_points
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        pts = _read_ply(self.files[i])
        pts = normalise(pts)
        if len(pts) >= self.num_points:
            idx = _fps(pts, self.num_points)
            pts = pts[idx]
        else:
            reps = int(np.ceil(self.num_points / len(pts)))
            pts = np.tile(pts, (reps, 1))[:self.num_points]
        # Shuffle point order to break any accidental dependencies (especially
        # important when tiling repeats points in deterministic order).
        np.random.shuffle(pts)
        if self.augment:
            theta = np.random.uniform(0, 2 * np.pi)
            R = np.array([[np.cos(theta), 0, np.sin(theta)],
                          [0, 1, 0],
                          [-np.sin(theta), 0, np.cos(theta)]], dtype=np.float32)
            pts = pts @ R
            pts = pts + np.random.normal(0, 0.02, pts.shape).astype(np.float32)
        pts = np.ascontiguousarray(pts, dtype=np.float32)
        return torch.from_numpy(pts).transpose(0, 1), int(self.labels[i])
```

## تقرير
```
[dataset]
  files:          <N>
  format:         <ply|pcd|xyz|npy>
  points_per_sample: <int>
  normalise:      centre + unit sphere
  sampling:       FPS | random
  augmentation:   <list>
```

## قواعد
- قم دائمًا بالتوسيط قبل القياس؛ يؤدي تبديل الترتيب إلى تغيير معنى "وحدة المجال".
- أفضّل FPS على أخذ العينات العشوائية لمهام الشكل؛ العشوائي جيد للتجزئة حيث تكون كل نقطة مهمة على أي حال.
- لا تزيد أبدًا أثناء التقييم؛ فقط أثناء التدريب.
- إذا كانت ملفات السحابة النقطية تشتمل على ألوان أو قيم عادية كقنوات إضافية، فقم بتوسيع مجموعة البيانات لإرجاع موتر `(3 + C, num_points)`، وليس xyz فقط.