---
name: skill-lora-training-setup
description: Write a full LoRA training config for a custom dataset, including captions, rank, batch size, and learning rate
version: 1.0.0
phase: 4
lesson: 11
tags: [computer-vision, stable-diffusion, lora, fine-tuning]
---

# LoRA Training Setup

قم بتحويل وصف الهدف الدقيق إلى تكوين تدريب ملموس وجاهز للانتقال إلى `diffusers` أو `kohya_ss`.

## When to use

- تدريب LoRA لموضوع (شخص، كائن، شخصية)، أسلوب (فنان، علامة تجارية)، أو مفهوم (وضعية، إضاءة).
- توسيع LoRA موجود بمزيد من البيانات.
- تصحيح أخطاء تشغيل LoRA الذي لا يتناسب مخرجاته مع صور التدريب أو يفوقها.

## Inputs

- `purpose`: موضوع | اسلوب | مفهوم
- `num_images`: كم عدد الصور التدريبية المتوفرة
- `base_model`: SD 1.5 | SDXL | SD3 | FLUX
- `gpu_vram_gb`: 8 | 12 | 16 | 24 | 48+
- `caption_source`: يدوي | BLIP2- متولدة | مجموعة البيانات الأصلية

## Rank picker

| الغرض | الرتبة | ألفا |
|---------|------|-------|
| الموضوع | 8-16 | رتبة |
| النمط | ١٦-٣٢ | المرتبة *2 |
| المفهوم | 32-64 | رتبة |

رتبة أعلى = سعة أكبر، ومخاطر أكثر للتجهيز على مجموعات البيانات الصغيرة. يقوم ألفا بقياس قوة تأثير LoRA؛ `alpha == rank` هو الإعداد الافتراضي الآمن. الأنماط هي الاستثناء الموثق: `alpha == rank * 2` يعطي دفعة أقوى للأسلوب على حساب المزيد من مخاطر إضفاء الطابع الصعب على النمط - استخدم فقط عندما لا تكون الدقة في الموضوع هي الهدف.

## Training step target

- `subject` مع 5-20 صورة: 500-1500 خطوة.
- `style` مع 30-100 صورة: 1500-4000 خطوة.
- `concept` بأكثر من 100 صورة: 4000-10000 خطوة.

التجاوز على مسؤوليتك الخاصة - LoRA الذي حفظ صور التدريب الخاصة به لا يمكنه التعميم.

## Learning rate

- تشفير النص LoRA: `1e-4` لـ SD 1.5، `5e-5` لـ SDXL.
- يو نت LoRA: `1e-4` بـ SD 1.5، `1e-4` بـ SDXL.
- FLUX / SD3: `5e-5` بالنسبة للمحول، عادةً ما يتم تجميد برامج ترميز النص.
- خفض LR إلى النصف عند `num_images < 15` (الموضوع) أو عند التدريب لأكثر من 3000 خطوة؛ تستفيد مجموعات البيانات الصغيرة والمدى الطويل من التحديث اللطيف.

## Scheduler

- `cosine_with_warmup` (افتراضي): الإحماء خلال أول 5-10% من الخطوات، ثم اضمحلال جيب التمام. استخدم عندما `steps >= 1000`؛ يعطي ذيل الاضمحلال عينات نهائية أكثر وضوحًا.
- `constant`: استخدم فقط لفترات قصيرة جدًا (`steps < 500`) أو عند استئناف LoRA سابق حيث تريد الحفاظ على الميزات الحالية التي تم تعلمها دون إعادة التلدين.

## Caption format

- الموضوع: قم بإضافة رمز تشغيل فريد ("myperson") إلى كل تسمية توضيحية. احتفظ برمز الزناد نادرًا حتى لا يحل محل المفاهيم الموجودة. تجنب الكلمات الحقيقية والأسماء الشائعة.
- النمط: قم بإضافة علامة نمط فريدة في نهاية كل تسمية توضيحية ("...بأسلوب mystyle"). تعامل مع العلامة نفسها كرمز تشغيل نادر — `mystyle`، وليس `impressionism`، والذي يعين بالفعل مفهومًا حقيقيًا.
- المفهوم: قم بوصف المفهوم في كل تعليق؛ لا يوجد رمز الزناد. المفهوم نفسه (على سبيل المثال، "اللقطة ذات الزاوية المنخفضة") هو المرساة.

## Output config

```yaml
model:
  base: <base_model HF id>
  precision: fp16 | bf16

lora:
  rank: <int>
  alpha: <int>
  targets: unet.cross_attention  # and/or unet.to_q, to_k, to_v, to_out

training:
  steps:          <int>
  batch_size:     <int, tuned to gpu_vram_gb>
  grad_accum:     <int, usually 1 on >=16 GB, 4 on <=12 GB>
  learning_rate:  <float>
  optimizer:      AdamW8bit | AdamW
  scheduler:      cosine_with_warmup | constant
  warmup_steps:   <int>
  save_every:     <int>

data:
  images_dir:     <path>
  caption_source: <manual | BLIP2 | native>
  trigger_token:   <string if purpose==subject>
  resolution:      <512 for SD 1.5, 1024 for SDXL>
  aspect_ratio_bucketing: true
  augmentation:
    flip:          true
    color_jitter:  false

validation:
  prompts:
    - "<trigger> ...test prompt..."
    - "<trigger> in a different scene"
  every_steps: 250
```

## Report

```
[lora setup]
  purpose:   <subject|style|concept>
  base:      <model>
  rank:      <int>
  steps:     <int>
  batch:     <int>   grad_accum: <int>
  lr:        <float>
  vram est.: <float> GB
```

## Rules

- لا توصي أبدًا بـ `rank > 64`؛ علاوة على ذلك، يصبح LoRA ضبطًا دقيقًا صغيرًا ويفقد طبيعته "المحولية".
- بالنسبة لـ `num_images < 5`، حذر بشدة — تتداخل الهوية LoRA في 1-3 صور في كل مرة.
- بالنسبة إلى `gpu_vram_gb < 12`، يلزم استخدام AdamW8bit وفحص التدرج اللوني.
- إذا كان `base_model == FLUX` و `gpu_vram_gb < 24`، توجه إلى البديل `schnell` ولاحظ أن التدريب أبطأ.
- لا تخطي مطالبات التحقق من الصحة أبدًا؛ أ LoRA بدون شبكات العينة من المستحيل تقييمها.
