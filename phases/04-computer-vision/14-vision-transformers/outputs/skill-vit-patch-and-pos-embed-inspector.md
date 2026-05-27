---
name: skill-vit-patch-and-pos-embed-inspector
description: Verify a ViT's patch embedding and positional embedding shapes match the model's expected sequence length
version: 1.0.0
phase: 4
lesson: 14
tags: [vision-transformer, debugging, pytorch]
---

# ViT Patch ومفتش التضمين الموضعي
الخطأ الأكثر شيوعًا في نقل ViT: تحميل نقطة تفتيش تم تدريبها مسبقًا على 224 × 224 في نموذج تم تكوينه لـ 384 × 384 (أو العكس). يحتوي التضمين الموضعي على طول تسلسل خاطئ وينتج النموذج القمامة بصمت.
##متى يستخدم
- ضبط ViT المُدرب مسبقًا بدقة غير افتراضية.
- التحقق من سبب فشل منفذ الوزن بين ViT-B/16 وViT-B/32؛ سيقوم المفتش بوضع علامة على عدم تطابق حجم التصحيح حتى يعرف المتصل تبديل البنيات بدلاً من فرض منفذ.
- تصحيح أخطاء ViT الذي يتم تحميله بدون أخطاء ولكنه يتدرب بشكل سيئ.
## المدخلات
- `model`: ViT تم إنشاء مثيل له `nn.Module`.
- `expected_image_size`: الارتفاع × العرض الذي سيشاهده النموذج في الإنتاج.
- `patch_size`: حجم التصحيح المتوقع.
## الخطوات
1. حدد موقع تحويل تضمين التصحيح داخل النموذج. قم بالإبلاغ عن `kernel_size`، `stride`، `in_channels`، `out_channels`.
2. حساب العدد المتوقع من التصحيحات. للصورة المربعة: `(image_size / patch_size)^2`. للمستطيل: `(H / patch_size) * (W / patch_size)`. يتطلب `H % patch_size == 0` و`W % patch_size == 0`؛ خلاف ذلك العلم والرفض.
3. حدد موقع التضمين الموضعي الذي تم تعلمه. الإبلاغ عن شكله `(1, N, dim)`.
4. قارن `N` بـ `num_patches + 1` (مع CLS) أو `num_patches` (بدون CLS). عدم التطابق يعني أنه تم تدريب نقطة التفتيش مسبقًا بدقة مختلفة أو حجم تصحيح مختلف.
5. تأكد من أن `out_channels` لتحويل التصحيح يساوي `dim` للتضمين الموضعي.
6. إذا كان من المفترض أن يقوم النموذج باستيفاء التضمينات الموضعية للقرارات الجديدة، فتحقق من وجود أداة الاستيفاء المساعدة (معظم `timm` ViTs تفعل ذلك تلقائيًا عبر `resize_pos_embed`).
## تقرير
```
[vit-inspector]
  image_size:         HxW
  patch_size:         <int>
  num_patches (computed): <int>
  patch_conv:         k=<int>  s=<int>  in=<int>  out=<int>
  pos_embed shape:    (1, N, dim)
  has CLS token:      yes | no
  pos_embed N:        <int>    expected: <int>
  verdict:            ok | mismatch

[if mismatch]
  action:  reinitialise pos_embed for new sequence length
  tool:    timm.models.vision_transformer.resize_pos_embed
```

## قواعد
- لا تقم أبدًا بالتحريف بصمت دون سابق إنذار؛ قم بإظهار الإجراء حتى يعرف المستخدم أن البنية الموضعية المُدربة مسبقًا ربما تكون قد تغيرت.
- في حالة عدم تطابق patch_size، ارفض التوصية بالاستيفاء — وقم بالتبديل إلى البنية الصحيحة.
- لا تحاول تثبيت النموذج في مكانه؛ تقرير واقتراح.