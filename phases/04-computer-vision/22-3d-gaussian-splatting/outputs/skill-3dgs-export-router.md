---
name: skill-3dgs-export-router
description: Pick the right 3DGS export format (.ply / .splat / glTF KHR_gaussian_splatting / USD) given the downstream viewer or engine
version: 1.0.0
phase: 4
lesson: 22
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# جهاز توجيه التصدير 3DGS
قم بتعيين الهدف النهائي إلى تنسيق ملف 3DGS الصحيح. يوفر ساعات من تصحيح الأخطاء "لا يتم التحميل".
##متى يستخدم
- بعد التدريب على مشهد 3DGS، قبل مشاركته مع محتوى pipeline.
- الاختيار بين تنسيقات درجة البحث (.ply) وتنسيقات درجة الإنتاج (glTF / USD).
- تسليم خط الأنابيب: فريق الالتقاط -> مهندس 3DGS -> مصمم الألعاب / VFX فنان / مطور ويب.
## المدخلات
- `target_engine`: غير واقعي | الوحدة | الكون | خلاط | Vision_pro | three_js | babylon_js | السيزيوم | بلايكانفاس | com.supersplat
- `priority`: قابلية النقل | حجم الملف | Quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3
## قرار التنسيق
| الهدف | التنسيق الموصى به | لماذا |
|--------|--------------------|-----|
| محرك غير واقعي (إنتاج افتراضي) | البرنامج المساعد Volinga أو glTF KHR_gaussian_splatting | مسار SDK غير واقعي أصلي |
| الوحدة (XR/لعبة) | .ply عبر البرنامج المساعد Aras-P Unity-GaussianSplatting | الوحدة المتوافقة مع معايير المجتمع pipeline |
| NVIDIA Omniverse، أدوات بيكسار | مفتوحUSD 26.03 (UsdVolParticleField3DGaussianSplat) | النوع الأساسي USD الأصلي |
| أبل فيجن برو | مفتوحUSD 26.03 | أصلي لنظام التشغيل VisionOS 2.x |
| خلاط | .ply + KIRI الوظيفة الإضافية للمحرك | الوظيفة الإضافية للمجتمع تقرأ البقع الأولية |
| عارض الويب Three.js | glTF KHR_gaussian_splatting أو .splat | معيار المتصفح، ويعمل مع `GaussianSplats3D` |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9 تمت إضافة الدعم الأصلي |
| السيزيوم (CesiumJS 1.139+، السيزيوم لـ Unreal 2.23+) | glTF KHR_gaussian_splatting | شحنها دعما صريحا |
| بلاي كانفاس | .سبلات | PlayCanvas التنسيق الكمي الأصلي |
| سوبر سبلات (محرر) | .ply أو .splat | استيراد + تصدير |
## مقايضات التكميم
- `.ply` الدقة الكاملة: أكبر ملف، بدون فقدان، لأي مشاهد.
- `.splat`: أصغر بمقدار 4x-8x، مع فقدان طفيف للجودة وفقًا لمعاملات SH3، ومعيار النظام البيئي PlayCanvas.
- glTF KHR: قابل للتكوين عبر EXT_meshopt_compression؛ الأصغر مع أعلى التوافق.
- USD: مضغوط بواسطة عبوة USDZ؛ الأصغر بالنسبة لخطوط Apple pipelines.
## تقرير الإخراج
```
[export plan]
  target:         <engine>
  format:         <name>
  sh degree:      <0|1|2|3>
  compression:    <none|meshopt|quantisation|usdz>
  expected size:  <MB>
  compatible with: <list of viewers>

[pipeline]
  1. source: <.ply from training>
  2. optional: SuperSplat cleanup pass
  3. convert: <tool + CLI or API call>
  4. package: <.gltf / .glb / .usd / .usdz / .splat / .ply>
  5. validate: <viewer sanity check>
```

## قواعد
- لا تقم أبدًا بإزالة معاملات SH3 بصمت - فهي تغير الانعكاسات المرآوية بشكل واضح.
- إذا كان `priority == file_size`، فيوصي بـ `.splat` أو glTF مع Meshopt؛ تحذير من فقدان الجودة.
- بالنسبة لمنصات Apple، تفضل USD / USDZ على glTF في 2026؛ USDZ يتمتع بدعم VisionOS من الدرجة الأولى.
- إذا كان دعم 3DGS للعارض المستهدف قياسيًا مسبقًا (ما قبل فبراير 2026)، فيوصي بـ `.ply` والمحمل المخصص للمشاهد؛ لن يتم التعرف على glTF بمعيار Khronos بعد.
- التحقق دائمًا من صحة الملف الذي تم تصديره في عارض واحد على الأقل قبل تسليمه؛ يحدث الفساد الصامت أثناء التكميم.