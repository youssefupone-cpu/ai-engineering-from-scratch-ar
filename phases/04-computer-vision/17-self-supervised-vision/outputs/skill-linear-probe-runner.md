---
name: skill-linear-probe-runner
description: Write the complete linear-probe evaluation for any frozen encoder and labelled dataset
version: 1.0.0
phase: 4
lesson: 17
tags: [self-supervised, evaluation, linear-probe, pytorch]
---

# Linear Probe Runner

قم بتقييم ميزات برنامج التشفير المجمد من خلال تدريب مصنف خطي واحد في الأعلى. التقييم القياسي لكل ورقة خاضعة للإشراف الذاتي.

## When to use

- مقارنة نقاط التفتيش الخاضعة للإشراف الذاتي.
- تتبع جودة الميزة على فترات ما قبل التدريب.
- تحديد ما إذا كان برنامج التشفير الذي تم تدريبه مسبقًا جيدًا بما يكفي لمهمة المصب دون ضبط دقيق.

## Inputs

- `encoder`: متجمد `nn.Module` إرجاع ميزة التعتيم الثابت لكل صورة.
- `feature_dim`: أبعاد مخرجات التشفير.
- `train_dataset`: مجموعة البيانات المسماة (image، class_id).
- `val_dataset`: مجموعة معلقة.
- `num_classes`: فئات المهام.
- `epochs`: عادةً 100 لمقياس ImageNet، و50 لمجموعات البيانات الأصغر.

## Steps

1. اضبط برنامج التشفير على وضع التقييم و`requires_grad=False` في كل معلمة.
2. قم باستخراج كل من مجموعات القطارات و val مرة واحدة. تخزين كمصفوفات numpy أو ملف معين للذاكرة.
3. قم بتدريب `nn.Linear(feature_dim, num_classes)` على الميزات المخزنة مؤقتًا باستخدام SGD + جدول جيب التمام.
4. المعلمات الفائقة القياسية: `lr=0.1`، `momentum=0.9`، `weight_decay=0`، `batch_size=1024`. المسبار الخطي حساس بشكل مدهش لـ `lr` — المسح إذا كانت الدقة ضعيفة.
5. قم بالإبلاغ عن دقة أعلى 1 في نهاية التدريب.

## Output template

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

def extract(encoder, loader, device="cpu"):
    encoder.eval()
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            f = encoder(x.to(device)).cpu()
            feats.append(f)
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def linear_probe(encoder, feature_dim, train_loader, val_loader,
                 num_classes, epochs=50, lr=0.1, device="cpu"):
    for p in encoder.parameters():
        p.requires_grad = False

    f_train, y_train = extract(encoder, train_loader, device)
    f_val, y_val = extract(encoder, val_loader, device)

    head = nn.Linear(feature_dim, num_classes).to(device)
    opt = SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    sched = CosineAnnealingLR(opt, T_max=epochs)

    ds = torch.utils.data.TensorDataset(f_train, y_train)
    train_iter = DataLoader(ds, batch_size=1024, shuffle=True)

    best_val = 0.0
    for ep in range(epochs):
        head.train()
        for x, y in train_iter:
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(head(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            acc = (head(f_val.to(device)).argmax(-1).cpu() == y_val).float().mean().item()
        best_val = max(best_val, acc)
    return best_val
```

## Report

```
[linear probe]
  encoder:     <name + pretrain checkpoint>
  feature_dim: <int>
  epochs:      <int>
  best_val_top1: <float>
```

## Rules

- لا تقم مطلقًا بتحديث أوزان جهاز التشفير أثناء المسبار الخطي؛ سيكون ذلك بمثابة ضبط دقيق وليس تحقيقًا.
- ميزات الحساب المسبق مرة واحدة؛ تؤدي إعادة تدريب برنامج التشفير في كل فترة إلى إهدار 100 ضعف من الحوسبة.
- استخدم SGD مع جدول جيب التمام وعدم فقدان الوزن؛ أداء آدم ضعيف هنا في بعض الأحيان.
- معدلات التعلم الاجتياحية مرة واحدة على الأقل لكل عائلة تشفير؛ يختلف الأمثل عبر طرق SSL.
