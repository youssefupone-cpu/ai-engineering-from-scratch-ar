---
name: skill-3dgs-export-router
description: Pick the right 3DGS export format (.ply / .splat / glTF KHR_gaussian_splatting / USD) given the downstream viewer or engine
version: 1.0.0
phase: 4
lesson: 22
tags: [3d-gaussian-splatting, export, glTF, OpenUSD, pipeline]
---

# 3DGS Export Router

قم بتعيين الهدف النهائي إلى تنسيق ملف 3DGS الصحيح. يوفر ساعات من تصحيح الأخطاء "لا يتم التحميل".

## When to use

- بعد التدريب على مشهد 3DGS، قبل مشاركته بمحتوى pipeline.
- الاختيار بين تنسيقات درجة البحث (.ply) وتنسيقات درجة الإنتاج (glTF / USD).
- تسليم خط الأنابيب: فريق الالتقاط -> مهندس 3DGS -> مصمم الألعاب / VFX فنان / مطور ويب.

## Inputs

- `target_engine`: غير حقيقي | الوحدة | الكون | خلاط | Vision_pro | three_js | babylon_js | السيزيوم | بلايكانفاس | com.supersplat
- `priority`: قابلية النقل | حجم الملف | Quality_preservation
- `include_sh_degree`: 0 | 1 | 2 | 3

## Format decision

| الهدف | التنسيق الموصى به | لماذا |
|--------|--------------------|-----|
| محرك غير واقعي (إنتاج افتراضي) | البرنامج المساعد Volinga أو glTF KHR_gaussian_splatting | مسار SDK أصلي غير واقعي |
| الوحدة (XR/لعبة) |.ply عبر البرنامج المساعد Aras-P Unity-GaussianSplatting | الوحدة بمعايير المجتمع pipeline |
| NVIDIA Omniverse، أدوات بيكسار | مفتوحUSD 26.03 (UsdVolParticleField3DGaussianSplat) | أصلي USD النوع الأساسي |
| أبل فيجن برو | مفتوحUSD 26.03 | أصلي لنظام التشغيل VisionOS 2.x |
| خلاط |.ply + KIRI إضافة المحرك | الوظيفة الإضافية للمجتمع تقرأ البقع الأولية |
| عارض الويب Three.js | glTF KHR_gaussian_splatting أو.splat | معيار المتصفح، يعمل مع `GaussianSplats3D` |
| Babylon.js V9+ | glTF KHR_gaussian_splatting | V9 تمت إضافة الدعم الأصلي |
| السيزيوم (CesiumJS 1.139+، السيزيوم لـ Unreal 2.23+) | glTF KHR_gaussian_splatting | شحنها دعما صريحا |
| بلاي كانفاس |.سبلات | PlayCanvas التنسيق الكمي الأصلي |
| سوبر سبلات (محرر) |.ply أو.splat | استيراد + تصدير |

## Quantisation trade-offs

- `.ply` الدقة الكاملة: أكبر ملف، بدون فقدان، لأي مشاهد.
- `.splat`: أصغر بمقدار 4x-8x، مع فقدان طفيف للجودة على معاملات SH3، معيار نظام PlayCanvas البيئي.
- glTF KHR: قابل للتكوين عبر EXT_meshopt_compression؛ الأصغر مع أعلى التوافق.
- USD: مضغوط بواسطة USDZ التغليف؛ الأصغر لأبل pipelines.

## Output report

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

## Rules

- لا تجرد مطلقًا معاملات SH3 بصمت - فهي تغير الانعكاسات المرآوية بشكل واضح.
- إذا كان `priority == file_size`، فيوصي بـ `.splat` أو glTF مع Meshopt؛ تحذير من فقدان الجودة.
- بالنسبة لمنصات Apple، تفضل USD / USDZ على glTF في 2026؛ USDZ يتمتع بدعم VisionOS من الدرجة الأولى.
- إذا كان دعم 3DGS للعارض المستهدف قياسيًا مسبقًا (ما قبل فبراير 2026)، فيوصي بـ `.ply` والمحمل المخصص للمشاهد؛ لن يتم التعرف على glTF بمعيار Khronos بعد.
- التحقق دائمًا من صحة الملف الذي تم تصديره في عارض واحد على الأقل قبل تسليمه؛ يحدث الفساد الصامت أثناء التكميم.
