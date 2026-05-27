---
name: mcp-handshake-tracer
description: Given a pcap-style transcript of an MCP client-server conversation, annotate every message with its primitive, lifecycle phase, and capability dependency.
version: 1.0.0
phase: 13
lesson: 06
tags: [mcp, json-rpc, lifecycle, capabilities]
---

بالنظر إلى تسلسل من JSON-RPC 2.0 مغلفات تم التقاطها من جلسة MCP، قم بإنتاج جولة تفصيلية تحدد المرحلة الأولية لكل رسالة ومرحلة دورة الحياة وعلامة القدرة الأساسية.

Produce:

1. شرح لكل رسالة. For each `{request, response, notification}`, state: direction (client-to-server or server-to-client), primitive (tools / resources / prompts / roots / sampling / elicitation / lifecycle), lifecycle phase, and the capability flag that had to be negotiated for this message to be valid.
2. فحص القدرة. أعد بناء التبادل `initialize` من النص وقم بإدراج كافة الإمكانات التي تم التفاوض عليها. ضع علامة على أي رسالة من شأنها أن تنتهك القدرة الغائبة.
3. تشخيص الأخطاء. لكل خطأ JSON-RPC، قم بتسمية الرمز والسبب الأكثر احتمالاً في ضوء السياق المحيط.
4. تدقيق الاكتمال. قم بوضع علامة على النص الذي يفتقد أحد: `initialize`، `initialized` إشعار، واحد على الأقل `tools/list` أو ما يعادله، إيقاف تشغيل أنيق.
5. الامتثال للمواصفات. تحقق من معلمات كل طلب مقابل الحد الأدنى لمجموعة الحقول الخاصة بمواصفات 2025-11-25. الإبلاغ عن الإغفالات.

الرفض الصارم:
- أي رسالة تستخدم طريقة خارج مجموعة المواصفات المسموح بها بدون البادئة `x-`.
- أي رسالة `sampling/createMessage` عندما لم يعلن العميل عن القدرة `sampling`.
- أي استدعاء قبل وصول `notifications/initialized`.

قواعد الرفض:
- إذا طُلب منك تدقيق نص من بروتوكول غير MCP، ارفض وأشر إلى A2A المواصفات (المرحلة 13 · 19) كبديل.
- إذا طُلب منك "إصلاح" النص، ارفض. هذه المهارة توضح؛ لا إعادة كتابة. تصحيح المسار من خلال التنفيذ SDK.

الإخراج: سطر واحد مشروح لكل رسالة بترتيب الوصول: `[phase/primitive/capability] <method or result shape>`. انتهي بملخص من ثلاثة أسطر يذكر أي انتهاكات للقدرة وأي خطوات مفقودة في دورة الحياة.
