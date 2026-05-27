---
name: voice-pipeline
description: Scaffold a Pipecat-shaped voice pipeline (VAD + STT + LLM + TTS + transport) with barge-in, confidence gating, and latency budget enforcement.
version: 1.0.0
phase: 14
lesson: 22
tags: [voice, pipecat, livekit, webrtc, latency]
---

نظرًا لمواصفات المنتج الصوتي (اللغة، النقل، مقدمي الخدمة)، قم ببناء خط قائم على الإطار pipe.

Produce:

1. اكتب `Frame` مع `kind`، `payload`، `direction` (المصب/المنبع).
2. المعالجات: `VAD`، `STT`، `LLM`، `TTS`، `Transport`. لكل منها `process(frame)`.
3. `link()` مساعد تسلسل المعالجات للأمام والخلف.
4. إلغاء معالجة الإطار: مسار UPSTREAM من النقل إلى TTS إلى LLM إلى STT، وإسقاط العمل المعلق في كل مرحلة.
5. المراقبون: مقاييس زمن الوصول لكل مرحلة؛ ينبعث نطاق OTel لكل إطار يعبر المعالج (الدرس 23).
6. بوابة الثقة على STT: أسفل العتبة، قم بإصدار إطار نصي "يرجى التكرار" بدلاً من النص.

الرفض الصارم:

- خط الأنابيب دون التعامل مع المنبع. المشاركة ليست اختيارية للصوت.
- LLM مكالمات بدون بث. يهيمن زمن الوصول للرمز الأول؛ يجب أن يتم دفقها.
- الثقة العمياء STT. يؤدي تقديم نصوص خاطئة إلى LLM إلى ردود خاطئة.

قواعد الرفض:

- إذا تجاوز زمن الوصول من طرف إلى طرف 1500 مللي ثانية في التشغيل البارد، فارفض الشحن. قم بتحسين السلسلة أو استخدم MultimodalAgent (LiveKit Direct-audio).
- إذا كان المنتج هاتفيًا أولاً وكان pipeline لا يحتوي على محول SIP، قم بالرفض. الطريق عبر LiveKit SIP أو منصة (Vapi/Retell).
- إذا كان المنتج يحمل PII صوتًا بدون تشفير أثناء النقل، ارفض.

الإخراج: `frames.py`، `processors.py`، `pipelineeline.py`، `observers.py`، `README.md` يشرح ميزانية زمن الوصول وتصميم البارجة واختيار النقل. اختتم بـ "ما يجب قراءته بعد ذلك" بالإشارة إلى الدرس 23 (OTel)، أو الدرس 24 (الواجهات الخلفية لإمكانية المراقبة)، أو مستندات LiveKit للحصول على تفاصيل WebRTC.
