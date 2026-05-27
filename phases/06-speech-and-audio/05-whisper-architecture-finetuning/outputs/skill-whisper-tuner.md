---
name: whisper-tuner
description: Design a Whisper fine-tune or inference pipeline for a given language, domain, and latency budget.
version: 1.0.0
phase: 6
lesson: 05
tags: [audio, whisper, asr, fine-tuning, lora]
---

بالنظر إلى الهدف (مجموعة اللغة، المجال، توزيع طول المقطع، ميزانية زمن الوصول، الأجهزة) والبيانات (الساعات المتاحة، الجودة)، الإخراج:
1. البديل. صغير / أساسي / صغير / متوسط ​​/ كبير - الإصدار 3 / توربو. سبب.
2. وقت التشغيل. الفانيليا / الهمس الأسرع / الهمس / تدفق الهمس. سبب.
3. خطة الضبط الدقيق. Full-FT vs LoRA (r, target_modules)، سياسة تشفير التجميد، عدد العصور.
4. حراس الاستدلال. VAD (Silero أو Whisper الخاص)، `temperature=0`، `condition_on_previous_text=False`، `no_speech_threshold`.
5. التقييم. هدف المجال WER، قواعد تطبيع النص، التحقق من معدل الهلوسة على مقاطع الصمت.
ارفض نشر Whisper على صوت عشوائي بدون VAD. ارفض تعيين `condition_on_previous_text=True` للمهام متعددة الأجزاء دون وجود حارس هارب. قم بوضع علامة على أي ضبط دقيق يقوم بتبديل رمز Whisper المميز أو mel pipeline.