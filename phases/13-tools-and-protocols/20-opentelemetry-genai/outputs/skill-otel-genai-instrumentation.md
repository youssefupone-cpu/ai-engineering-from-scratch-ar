---
name: otel-genai-instrumentation
description: Produce an instrumentation plan for an agent codebase to emit OTel GenAI spans end-to-end.
version: 1.0.0
phase: 13
lesson: 19
tags: [otel, observability, gen-ai, tracing]
---

بالنظر إلى قاعدة تعليمات الوكيل (استدعاءات LLM، إرسال الأداة، عميل MCP، الوكلاء الفرعيين)، قم بإنتاج خطة أدوات OTel GenAI.
ينتج:
1. يمتد التسلسل الهرمي. الجذر `agent.invoke_agent` (داخلي) والأطفال: `llm.chat` (CLIENT)، `tool.execute` (داخلي)، `mcp.call` (CLIENT)، `subagent.invoke` (داخلي).
2. قائمة مراجعة السمات لكل فترة. `gen_ai.operation.name`, `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.response.model`, `gen_ai.usage.*`, `gen_ai.tool.name`, `gen_ai.agent.name`.
3. قاعدة الانتشار. أدخل W3C أثر التتبع في كل مكالمة عن بعد؛ بالنسبة لـ MCP استخدم stdio `_meta.traceparent` كحقل مؤقت.
4. سياسة التقاط المحتوى. معطل بشكل افتراضي؛ المستند الذي يمكّنه env var؛ اسم PII المخاطر.
5. اختيار المصدر. جايجر / تيمبو / لانجفيوز / فينيكس / داتا دوج / قرص العسل؛ OTLP كالسلك.
الرفض الصارم:
- أي خطة تفتقد إلى انتشار التتبع عبر MCP أو حدود الوكيل الفرعي.
- أي خطة يتم فيها تشغيل التقاط المحتوى بشكل افتراضي. مطالبات التسريبات و PII.
- أي خطة تصدر سمات مخصصة عشوائية بدون `gen_ai.` أو بادئة البائع الصريحة.
قواعد الرفض:
- إذا كانت قاعدة التعليمات البرمجية تستخدم إطار عمل مزودًا بأدوات OTel التلقائية المضمنة (Pydantic AI، LangGraph، AgentOps)، فيوصي بربط إطار العمل أولاً.
- إذا كانت الواجهة الخلفية للمصدر محلية ولم يكن لدى الفريق دعم SRE، فاقترح استخدام واجهة خلفية مُدارة.
- إذا طلب المستخدم التقاط محتوى لتصحيح أخطاء المنتج، فارفض بدون سياسة موافقة مكتوبة وPII تنقيح pipeline.
الإخراج: خطة من صفحة واحدة مع تسلسل هرمي للامتداد، وقائمة مراجعة السمات لكل امتداد، وقاعدة النشر، وسياسة التقاط المحتوى، واختيار المصدر. انتهي بالمقياس العلوي للتنبيه عليه (عادة p95 `gen_ai.client.operation.duration`).