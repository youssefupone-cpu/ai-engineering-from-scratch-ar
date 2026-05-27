---
name: skill-dcgan-scaffold
description: Write a complete DCGAN scaffold from z_dim, image_size, and num_channels, including training loop and sample saver
version: 1.0.0
phase: 4
lesson: 9
tags: [computer-vision, gan, dcgan, scaffolding]
---

# DCGAN سقالة
في ضوء ثلاث معلمات، قم بإصدار هيكل مشروع قابل للتشغيل DCGAN مع حجم البنية بشكل صحيح لدقة الصورة المستهدفة.
##متى يستخدم
- بدء تجربة توليدية جديدة على مجموعة بيانات صغيرة.
- تدريس أساسيات DCGAN مع الحد الأدنى من الأمثلة العملية.
- النماذج الأولية لشبكات GAN الشرطية (يتم حقن التسمية في نفس السقالة).
## المدخلات
- `image_size`: واحد من 32، 64، 128 (يجب أن يكون قوة لاثنين).
- `num_channels`: 1 (تدرج رمادي) أو 3 (RGB).
- `z_dim`: عادةً 64 أو 128.
- `with_spectral_norm`: نعم | لا؛ الافتراضي نعم.
## تحجيم العمارة
يعتمد عدد كتل التحويل المنقولة في G وكتل التحويل المقسمة في D على `image_size`:
| image_size | كتل G | كتل د |
|------------|----------|----------|
| 32 | 4 | 4 |
| 64 | 5 | 5 |
| 128 | 6 | 6 |
كل كتلة إضافية تضاعف (G) أو تنصف (D) البعد المكاني. يبدأ عدد الميزات عند 32 ويتدرج مع `feat_base * 2^block_index`.
## ملفات الإخراج
- `model.py` — فئات المولد + التمييز
- `train.py` — حلقة التدريب، الخسارة، إعداد المحسن
- `sample.py` — نموذج لحافظة الشبكة
- `config.json` — المعلمات الفائقة
- `README.md` — بداية سريعة مكونة من 10 أسطر
## تقرير
```
[scaffold]
  image_size:       <int>
  num_channels:     <int>
  z_dim:            <int>
  spectral_norm:    yes | no

[arch]
  G blocks:         <N>, channels: [list]
  D blocks:         <N>, channels: [list]
  G params (est):   <N>
  D params (est):   <N>

[training defaults]
  optimizer:   Adam(lr=2e-4, betas=(0.5, 0.999))
  batch_size:  64
  epochs:      50
  sample_every: 1 epoch

[files written]
  - model.py
  - train.py
  - sample.py
  - config.json
  - README.md
```

## قواعد
- استخدم دائمًا `nn.Tanh()` في مخرجات G وقم بقياس البيانات إلى [-1، 1] أثناء التدريب.
- استخدم دائمًا `LeakyReLU(0.2)` في D.
- عندما `with_spectral_norm == yes`، قم بتغليف كل تحويل في D باستخدام `spectral_norm()` وقم بإزالة BatchNorm من D. احتفظ بـ BatchNorm في G.
- لا تطلق سقالة أبدًا لـ image_size > 128 — يصبح DCGAN غير مستقر فوق ذلك؛ قم بتوجيه المستخدم إلى StyleGAN أو نموذج الانتشار.