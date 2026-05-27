---
name: skill-pytorch-patterns
description: Reference patterns for PyTorch training, evaluation, and deployment
version: 1.0.0
phase: 03
lesson: 11
tags: [pytorch, training, deep-learning, gpu, patterns]
---

## Canonical Training Loop

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
```

## Mixed Precision Training

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)
    optimizer.zero_grad()
    with autocast(device_type="cuda"):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

استخدم عندما: التدريب على GPU باستخدام الأجهزة القادرة على استخدام float16 (V100، A100، H100، RTX 3090+). توقع تسريع ~1.5-2x وتقليل الذاكرة بنسبة ~50%.

## Gradient Accumulation

```python
accumulation_steps = 4
optimizer.zero_grad()
for i, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

يُستخدم عندما: يجب أن يكون حجم الدفعة الفعال أكبر من GPU الذي تسمح به الذاكرة. يؤدي قسمة الخسارة على خطوات التراكم إلى الحفاظ على اتساق مقياس التدرج.

## Save and Load

```python
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": loss.item(),
}, "checkpoint.pt")

checkpoint = torch.load("checkpoint.pt", weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

قم دائمًا بحفظ حالة المحسن لاستئناف التدريب. للاستدلال فقط، احفظ فقط `model.state_dict()`.

## Custom Dataset

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = self._load_samples(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def _load_samples(self, data_dir):
        ...
```

## DataLoader Configuration

```python
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
)
```

| المعلمة | ماذا يفعل | متى تستخدم |
|-----------|-------------|-------------|
| num_workers=4 | تحميل البيانات الموازية | دائمًا على الأجهزة متعددة النواة |
| pin_memory=صحيح | صفحة مقفلة CPU الذاكرة | عند التدريب على GPU |
| drop_last=صحيح | إسقاط الدفعة النهائية غير مكتملة | عند استخدام BatchNorm |
| Constant_workers=صحيح | إبقاء العمال على قيد الحياة عبر العصور | عندما يكون num_workers > 0 |

## Learning Rate Schedules

```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=num_epochs * len(train_loader),
    pct_start=0.1,
)

for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        ...
        optimizer.step()
        scheduler.step()
```

OneCycleLR: أفضل خيار افتراضي لمعظم المهام. يسخن حتى max_lr، ثم يضمحل جيب التمام. اتصل بالرقم `scheduler.step()` بعد كل دفعة، وليس كل فترة.

## Weight Initialization

```python
def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

model.apply(init_weights)
```

## Inference Mode

```python
model.eval()

with torch.inference_mode():
    outputs = model(inputs)
```

`torch.inference_mode()` أسرع من `torch.no_grad()` لأنه يعطل الترقية التلقائية بالكامل بدلاً من مجرد منع حساب التدرج.

## Common Mistakes Checklist

1. تطبيق softmax قبل CrossEntropyLoss (يتضمن log_softmax داخليًا)
2. نسيان استدعاء model.eval() أثناء التحقق من الصحة
3. نسيان نقل الموترات إلى نفس جهاز النموذج
4. عدم استدعاءOptimer.zero_grad() (تتراكم التدرجات بشكل افتراضي)
5. استخدام torch.no_grad() أثناء التدريب (تعطيل حساب التدرج)
6. تعيين عدد كبير جدًا من num_workers (يؤدي إلى توليد عدد كبير جدًا من العمليات، ويدمر الذاكرة)
7. عدم استخدام pin_memory=True عند التدريب على GPU
8. حفظ كائن النموذج بالكامل بدلاً منstate_dict (فواصل عند إعادة البناء)
