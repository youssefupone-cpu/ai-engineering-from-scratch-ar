---
name: skill-noise-schedule-designer
description: Produce a linear, cosine, or sigmoid beta schedule given T and target corruption level, plus SNR plot
version: 1.0.0
phase: 4
lesson: 10
tags: [computer-vision, diffusion, noise-schedule, training]
---

# مصمم جدول الضوضاء
يتحكم جدول بيتا في مقدار الإشارة التي يتم الاحتفاظ بها في كل خطوة نشر. تؤدي الجداول الزمنية الضعيفة إلى الحد من كفاءة التدريب وجودة العينة في كل قرار نهائي.
##متى يستخدم
- بدء تشغيل تدريبي جديد للانتشار واختيار T والبيتا.
- تصحيح أخطاء نموذج الانتشار الذي ينتج عينات غير واضحة (الجدول الزمني شديد العدوانية) أو يفشل في تعلم البنية (الجدول الزمني معتدل للغاية).
- مقارنة التصاميم عبر الأوراق التي تشير إلى جداول زمنية مختلفة.
## المدخلات
- `T`: عدد الخطوات الزمنية، عادةً 100-1000.
- `type`: خطي | جيب التمام | السيني.
- `target_alpha_bar_final`: جزء الإشارة الذي يجب الاحتفاظ به عند t=T، الافتراضي 0.001 (99.9% تالف).
- اختياري `image_resolution` — تستفيد الصور الأكبر حجمًا من الجداول التي تفسد بشكل أبطأ (جداول جيب التمام أو الجداول المتغيرة).
## جدول الصيغ
### خطي```
beta_t = beta_start + (beta_end - beta_start) * (t - 1) / (T - 1)
```
الإعدادات الافتراضية: beta_start=1e-4، beta_end=0.02 (DDPM ورقة).
### جيب التمام (نيكول وداريوال، 2021)```
alpha_bar_t = cos^2((t/T + s) / (1 + s) * pi/2)
beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
```
ق = 0.008. يحافظ على الإشارة لفترة أطول؛ أفضل في عدد الخطوات المنخفضة.
### السيني```
alpha_bar_t = 1 / (1 + exp(k * (t/T - 0.5)))
```
ك = 6 إلى 12. حل وسط جيد؛ يتم استخدامه بواسطة بعض متغيرات SDXL.
## الخطوات
1. حساب البيتا لكل صيغة.
2. الحساب المسبق `alphas`، `alphas_cumprod`، `sqrt_alphas_cumprod`، `sqrt_one_minus_alphas_cumprod`.
3. حساب SNR_t = alpha_bar_t / (1 - alpha_bar_t)؛ إنتاج ملخص SNR بمرور الوقت.
4. تحقق من أن `alphas_cumprod[T-1]` يقع ضمن 10% من `target_alpha_bar_final`؛ وإلا قم بضبط beta_end (الخطي)، أو s (جيب التمام)، أو k (السيني) وأعد المحاولة.
5. الإبلاغ عن ثلاث نقاط تفتيش:
   - `t=T*0.25` — الفساد المبكر
   - `t=T*0.5` — في منتصف الطريق
   - `t=T*0.75` — شبه نهائي
## تقرير
```
[schedule]
  type:   <name>
  T:      <int>
  beta_start: <float>   beta_end: <float>

[signal retention]
  t=0.25T:  alpha_bar=<X>  SNR=<X>
  t=0.5T:   alpha_bar=<X>  SNR=<X>
  t=0.75T:  alpha_bar=<X>  SNR=<X>
  t=T:      alpha_bar=<X>  SNR=<X>

[warnings]
  - <if alpha_bar collapses before 0.75T>
  - <if beta_end produces NaN in log-SNR>
```

## قواعد
- لا ترسل أبدًا جدولًا يحتوي على أي `alpha_bar_t <= 0`؛ قيم المشبك تحت 1e-5 وتحذير.
- جيب التمام هو التوصية الافتراضية لأخذ العينات ذات عدد الخطوات المنخفض (<30 خطوة).
- الخطي هو الإعداد الافتراضي لـ `quality_target == research` — يتم الإبلاغ عن خطوط الأساس DDPM بجداول خطية.
- عندما `image_resolution > 256`، يوصى بتغيير الجدول (تشن، 2023) للاحتفاظ بمزيد من الإشارة بدقة عالية.