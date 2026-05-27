---
name: mcp-auth-iii-wiring
description: Wire production MCP authorization (RFC 8414, 7591, 8707, 7636 PKCE, 9728) onto iii primitives — registerTrigger for HTTP/cron, registerFunction for validation, state::* for JWKS cache.
version: 1.0.0
phase: 13
lesson: 18
tags: [mcp, oauth, dcr, jwks, iii, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728]
---

بالنظر إلى تكوين الخادم MCP ومجموعة إمكانات IdP، قم بإصدار الأوليات iii وقواعد الرفض التي تشكل سطح مصادقة الإنتاج.

Inputs:

- `mcp_resource_url` — المورد الأساسي URL (بدون مسار)، يُستخدم كقيمة `aud` وكقيمة بيانات تعريف المورد المحمي `resource`.
- `idp_metadata_url` — مُوفِّر الهوية `/.well-known/oauth-authorization-server` URL.
- `idp_capabilities` — القيم التي تمت ملاحظتها لـ `code_challenge_methods_supported`، `grant_types_supported`، `registration_endpoint`، `response_types_supported`.
- `tools` — قائمة الأدوات MCP مع النطاق الذي يتطلبه كل منها.

Produce:

1. **بوابة الرفض.** إذا فشل أي من الشروط الأربعة، ارفض التوصيل وتوقف: - `` مفقود من `code_challenge_methods_supported`. - `authorization_code` مفقود من `grant_types_supported`. - `registration_endpoint` غائب (رقم RFC 7591 DCR). - `response_types_supported` هو أي شيء آخر غير `["code"]` بالضبط.

2. **وثيقة بيانات تعريف الموارد المحمية** (RFC 9728) لخادم MCP للنشر على `/.well-known/oauth-protected-resource`. تتضمن `resource`، `authorization_servers` (القائمة المسموح بها لجهة الإصدار)، `scopes_supported`، `bearer_methods_supported: ["header"]`.

3. **3 تشغيل التسجيلات.** قم بإصدار كل مكالمة حرفيًا: - `iii.registerTrigger("http", {"path": "/.well-known/oauth-protected-resource", "method": ""}, "auth::serve-protected-resource")` - `iii.registerTrigger("http", {"path": "/mcp", "method": ""}, "mcp::dispatch")` — يقوم المرسل باستدعاء `iii.trigger("auth::validate-jwt",...)` قبل تشغيل أي أداة. - `iii.registerTrigger("cron", {"schedule": "<rotation_schedule>"}, "auth::rotate-jwks")` — الجدول هو `0 */6 * * *` بشكل افتراضي؛ قم بالتشديد إلى `*/15 * * * *` لموفري الهوية ذوي التناوب العالي.

4. **3 تسجيلات الوظائف.** قم بإصدار كل مكالمة حرفيًا: - `iii.registerFunction("auth::validate-jwt", handler)` — التحقق من `iss` قائمة السماح، التوقيع مقابل التخزين المؤقت JWKS، `aud == mcp_resource_url`، `exp`، النطاق المطلوب. - `iii.registerFunction("auth::rotate-jwks", handler)` — جلب `jwks_uri`، يكتب `state::set("auth/jwks/<iss>", {keys, fetched_at})`. - `iii.registerFunction("auth::serve-protected-resource", handler)` — إرجاع المستند من (2). - `iii.registerFunction("auth::issue-step-up", handler)` — فقط إذا كانت قائمة الأدوات تحتوي على عمليات مسورة خلف نطاق لم يمنحه المستخدم في البداية.

5. **خطة مفتاح الدولة.** مفتاح واحد لكل جهة إصدار مقبولة: `auth/jwks/<issuer>` عقد `{keys, fetched_at}`. قم بتوثيق نمط القراءة: يقرأ المدقق من `state::get`، ويعود إلى خطأ متزامن `iii.trigger("auth::rotate-jwks",...)` في `kid`.

6. ** تعيين النطاق. ** قم بتعيين كل أداة إلى النطاق الذي تتطلبه. إخراج جدول: `| tool | required_scope | rationale |`. تجميع الأدوات التدميرية ضمن نطاقها الخاص؛ لا تقم أبدًا بإعادة استخدام نطاق القراءة لأداة الكتابة.

7. **قواعد الرفض في وقت التشغيل** (يجب على أداة التحقق تشفيرها وإصدارها في نص المعالج): - الرفض عند `aud!= mcp_resource_url`. - الرفض عند `iss not in authorization_servers`. - رفض عندما لا يكون `kid` في ذاكرة التخزين المؤقت JWKS بعد تراجع دورة واحدة. - رفض عند غياب النطاق المطلوب ← 403 `Bearer error="insufficient_scope", scope="<required>", resource="<mcp_resource_url>"`. - رفض أي طلب رمزي بدون المعلمة `code_verifier` أو `resource`.

الرفض الصارم (لا تقم أبدًا بإرسال أي من هذه الأشياء - ارفض الطلب وقم بتوثيق السبب):

- تخزين `client_secret` بنص عادي في متجر الحالة iii. يستخدم العملاء العامون `token_endpoint_auth_method: none`; يستخدم العملاء السريون `private_key_jwt`. لا توجد أسرار مشتركة بنص عادي في `state::*` أو في سجلات استجابة التسجيل.
- تخطي علامة `aud` في جهاز التحقق. النائب الحائر هو السبب الكامل لـ RFC 8707 + RFC 9728.
- السماح بطلبات رمز الترخيص PKCE-أقل. OAuth 2.1 يحظر ذلك؛ يجب أن يرفض المدقق أي تبادل `/token` يفتقر سجل رمز التفويض المخزن فيه إلى `code_challenge`.
- التخزين المؤقت JWKS بدون مهمة التحديث. إما أن يتم تشغيل سفن تشغيل cron، أو أن سطح المصادقة لا يتم نشره.
- الوثوق بالمطالبة `iss` بدون قائمة السماح. أي أداة تحقق تقبل رمزًا مميزًا من أي `iss` تتيح للمهاجم إنشاء موفر الهوية (IdP) الخاص به وتزوير الرموز المميزة.
- تخزين `registration_access_token` في نص عادي. التجزئة في الراحة. تتطلب نصًا واضحًا في كل تحديث.

الإخراج: خطة توصيل من صفحة واحدة مع مستند المورد المحمي، والاستدعاءات الثلاثة `registerTrigger`، والمكالمات الأربعة `registerFunction`، وخطة مفتاح الحالة، وجدول تعيين النطاق، وقواعد رفض وقت التشغيل المشفرة. انتهي بفجوة حظر النشر الفردية التي من المرجح أن تظهر أمام IdP المختار - عادةً DCR التوفر للمؤسسة SSO.
