---
name: skill-rectified-flow-trainer
description: Write a complete rectified-flow training loop with AdaLN DiT and Euler sampling
version: 1.0.0
phase: 4
lesson: 23
tags: [diffusion, rectified-flow, DiT, training]
---

# Rectified Flow Trainer

قم بإنتاج حلقة تدريب نظيفة وبسيطة يمكنها بنجاح تدريب DiT صغير مع تدفق مصحح على أي مجموعة بيانات موتر الصورة.

## When to use

- إعادة إنتاج الهدف التدريبي SD3 / FLUX على نطاق صغير.
- قياس التدفق المصحح مقابل DDPM على نفس البيانات.
- بناء نموذج التدفق المصحح المخصص للمجال غير القياسي (الطبي، الفضائي).

## Inputs

- `model`: `nn.Module` أخذ `(x, t)` وإرجاع السرعة المتوقعة.
- `dataset`: تكرار للصور النظيفة في مجال النموذج.
- `optimizer`: AdamW مع `lr=1e-4`، `weight_decay=0.01`، `betas=(0.9, 0.99)`.
- `scheduler`: جيب التمام مع عملية إحماء، افتراضي 1000 خطوة إحماء.

## Training step

```python
def rectified_flow_train_step(model, x0, optimizer, device):
    model.train()
    x0 = x0.to(device)
    n = x0.size(0)
    t = torch.rand(n, device=device)                     # uniform in [0, 1]
    epsilon = torch.randn_like(x0)
    x_t = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * epsilon
    target_v = epsilon - x0                              # velocity target
    pred_v = model(x_t, t)
    loss = F.mse_loss(pred_v, target_v)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
```

## Sampling (Euler)

```python
@torch.no_grad()
def sample(model, shape, steps=20, device="cpu"):
    model.eval()
    x = torch.randn(shape, device=device)
    dt = 1.0 / steps
    t = torch.ones(shape[0], device=device)
    for _ in range(steps):
        v = model(x, t)
        x = x - dt * v
        t = t - dt
    return x
```

## Tips

- استخدم `torch.rand` الزي `t`؛ يساعد أخذ العينات المرجحة logit-العادي أو نمط Sd3 لـ `t` قليلاً ولكن ليس مطلوبًا للبدء.
- EMA من الأوزان النموذجية هي الممارسة القياسية؛ الحفاظ على `ema_model` مع الاضمحلال 0.9999.
- إرشادات خالية من المصنفات للنماذج الشرطية: مع احتمال 10%، استبدل التكييف بتضمين فارغ/فارغ أثناء التدريب؛ عند الاستدلال، امزج `v_uncond + w * (v_cond - v_uncond)` مع `w` حول 3-5.
- للتدريب بأسلوب LDM (FLUX، SD3)، تعمل الحلقة بأكملها في مساحة كامنة VAE؛ النظيفة `x0` أعلاه هي في الواقع `VAE.encode(image)`.
- التقارب النموذجي على مجموعة بيانات لعبة 32 × 32: 2000-5000 خطوة. على التدريب الكامن SD3 الحقيقي: مئات الآلاف.

## Report

```
[rectified flow training]
  steps:        <int>
  final loss:   <float>
  ema decay:    <float>
  vae?:         yes | no
  cfg dropout:  <fraction>

[sampling]
  default steps: 20
  schnell / turbo target: 4
  full quality reference: 50+ (for comparison only)
```

## Rules

- لا تقم مطلقًا بتدريب التدفق المصحح باستخدام هدف سرعة مساحة الصورة على البيانات RGB `uint8`؛ التطبيع إلى متوسط ​​الصفر، تباين الوحدة أولاً.
- قم دائمًا بتسجيل خسارة التدريب لكل مجموعة زمنية؛ إذا كانت الخطوات الزمنية المبكرة (بالقرب من 0) بها خسارة أعلى من تلك المتأخرة (بالقرب من 1)، فمن المحتمل أن تكون معلمات السرعة خاطئة.
- لا تخلط هدف سرعة التدفق المصحح مع هدف الضوضاء DDPM في نفس حلقة التدريب؛ اختر واحدة.
- استخدم تدريب bfloat16 على Ampere+ GPUs؛ ينتج float16 أحيانًا خريجي NaN في التدفق المصحح بسبب حجم السرعة.
