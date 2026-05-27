---
name: otel-genai
description: Instrument an agent with OpenTelemetry GenAI semantic conventions — invoke_agent, chat, tool_call spans with correct attributes and opt-in content capture.
version: 1.0.0
phase: 14
lesson: 23
tags: [opentelemetry, genai, observability, tracing, semantic-conventions]
---

نظرًا لوقت تشغيل الوكيل، قم بتوصيل اصطلاحات OTel GenAI الدلالية.
ينتج:
1. `invoke_agent` مدى لكل تشغيل وكيل. النوع CLIENT لخدمات الوكيل عن بعد، وداخلي للخدمات قيد المعالجة. الاسم: `invoke_agent {gen_ai.agent.name}`.
2. `chat` مدى لكل LLM مكالمة مع `gen_ai.operation.name=chat`، `gen_ai.provider.name`، `gen_ai.request.model`، `gen_ai.response.model`.
3. `tool_call` مدى لكل استدعاء أداة باستخدام `gen_ai.tool.name` و، عند الاقتضاء، `gen_ai.data_source.id` (RAG مجموعة البيانات/مخزن الذاكرة).
4. التقاط محتوى الاشتراك: الافتراضي OFF؛ عند ON، قم بتخزين المدخلات/المخرجات خارجيًا وتسجيل `*.reference_id` على امتدادات.
5. نشر السياق: استخدم رؤوس سياق التتبع W3C بحيث يتم تشغيل العمليات المتعددة (Claude Agent SDK CLI عملية فرعية) في تتبع واحد.
الرفض الصارم:
- التقاط المطالبات/المخرجات الكاملة بشكل افتراضي. PII ومخاطر التسرب السري؛ كما ينتهك المواصفات.
- `gen_ai.provider.name` مفقود. تعطل لوحات المعلومات متعددة الموفرين.
- امتدادات الأداة اليتيمة. قم دائمًا بتعيين العلاقة بين الوالدين والطفل عبر السياق النشط.
قواعد الرفض:
- إذا لم يتمكن وقت التشغيل من نشر السياق عبر حدود العملية، فارفض. مطلوب خياطة تتبع متعددة العمليات لمستخدمي Claude Agent SDK + CLI.
- إذا كان المنتج يحتوي على قيود تنظيمية (HIPAA، GDPR)، فارفض التقاط المحتوى المضمن. مخزن خارجي مع التحكم في الوصول فقط.
- إذا لم يتم تعيين الواجهة الخلفية `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`، تحذير: قد تتغير أسماء السمات عند ترقية المجمع.
الإخراج: `tracer.py`، `attributes.py`، `content_store.py`، `README.md` شرح بنية النطاق واشتراك الاستقرار وسياسة التقاط المحتوى. اختتم بـ "ما يجب قراءته بعد ذلك" بالإشارة إلى الدرس 24 (الواجهات الخلفية: Langfuse، Phoenix، Opik) أو الدرس 17 لـ Claude Agent SDK نشر سياق التتبع.