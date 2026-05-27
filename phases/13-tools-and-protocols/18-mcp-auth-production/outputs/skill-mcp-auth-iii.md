---
name: mcp-auth-iii-wiring
description: Wire production MCP authorization (RFC 8414, 7591, 8707, 7636 PKCE, 9728) onto iii primitives — registerTrigger for HTTP/cron, registerFunction for validation, state::* for JWKS cache.
version: 1.0.0
phase: 13
lesson: 18
tags: [mcp, oauth, dcr, jwks, iii, rfc8414, rfc7591, rfc8707, rfc7636, rfc9728]
---

بالنظر إلى تكوين خادم MCP ومجموعة إمكانات IdP، قم بإصدار قواعد iii الأولية وقواعد الرفض التي تشكل سطح مصادقة الإنتاج.
المدخلات:
- `mcp_resource_url` — المورد الأساسي URL (بدون مسار)، يُستخدم كقيمة `aud` وكقيمة بيانات تعريف المورد المحمي `resource`.
- `idp_metadata_url` — `/.well-known/oauth-authorization-server` URL لموفر الهوية.
- `idp_capabilities` — القيم التي تمت ملاحظتها لـ `code_challenge_methods_supported`، `grant_types_supported`، ​​`registration_endpoint`، `response_types_supported`.
- `tools` — قائمة الأدوات MCP مع النطاق الذي يتطلبه كل منها.
ينتج:
1. **بوابة الرفض.** إذا فشل أي من الشروط الأربعة، ارفض التوصيل وتوقف:
   - `S256` مفقود من `code_challenge_methods_supported`.
   - `authorization_code` مفقود من `grant_types_supported`.
   - `registration_endpoint` غائب (رقم RFC 7591 DCR).
   - `response_types_supported` هو أي شيء آخر غير `["code"]` بالضبط.
2. **مستند بيانات تعريف الموارد المحمية** (RFC 9728) لخادم MCP للنشر في `/.well-known/oauth-protected-resource`. يتضمن `resource`، `authorization_servers` (القائمة المسموح بها لجهة الإصدار)، `scopes_supported`، `bearer_methods_supported: ["header"]`.
3. **3 تشغيل التسجيلات.** قم بإصدار كل مكالمة حرفيًا:
   - __الكود_0__
   - `iii.registerTrigger("http", {"path": "/mcp", "method": "POST"}, "mcp::dispatch")` — يقوم المرسل باستدعاء `iii.trigger("auth::validate-jwt", ...)` قبل تشغيل أي أداة.
   - `iii.registerTrigger("cron", {"schedule": "<rotation_schedule>"}, "auth::rotate-jwks")` — الجدول الزمني هو `0 */6 * * *` بشكل افتراضي؛ قم بالتشديد إلى `*/15 * * * *` لموفري الهوية ذوي التناوب العالي.
4. **3 تسجيلات الوظائف.** قم بإصدار كل مكالمة حرفيًا:
   - `iii.registerFunction("auth::validate-jwt", handler)` — يقوم بالتحقق من القائمة المسموح بها `iss`، والتوقيع مقابل JWKS المخزنة مؤقتًا، و`aud == mcp_resource_url`، و`exp`، والنطاق المطلوب.
   - `iii.registerFunction("auth::rotate-jwks", handler)` — يجلب `jwks_uri` ويكتب `state::set("auth/jwks/<iss>", {keys, fetched_at})`.
   - `iii.registerFunction("auth::serve-protected-resource", handler)` — يقوم بإرجاع المستند من (2).
   - `iii.registerFunction("auth::issue-step-up", handler)` — فقط إذا كانت قائمة الأدوات تحتوي على عمليات مسورة خلف نطاق لم يمنحه المستخدم في البداية.
5. **خطة مفتاح الحالة.** مفتاح واحد لكل جهة إصدار مقبولة: `auth/jwks/<issuer>` يحمل `{keys, fetched_at}`. قم بتوثيق نمط القراءة: يقرأ المدقق من `state::get`، ويعود إلى `iii.trigger("auth::rotate-jwks", ...)` المتزامن عند `kid` Miss.
6. ** تعيين النطاق. ** قم بتعيين كل أداة إلى النطاق الذي تتطلبه. إخراج جدول:
   __الكود_0__. تجميع الأدوات التدميرية ضمن نطاقها الخاص؛ لا تقم أبدًا بإعادة استخدام نطاق القراءة لأداة الكتابة.
7. **قواعد الرفض في وقت التشغيل** (يجب على أداة التحقق تشفيرها وإصدارها في نص المعالج):
   - الرفض عند `aud != mcp_resource_url`.
   - الرفض عند `iss not in authorization_servers`.
   - يتم الرفض عندما لا يكون `kid` موجودًا في ذاكرة التخزين المؤقت JWKS بعد تراجع دورة واحدة.
   - قم بالرفض عند غياب النطاق المطلوب → 403 `Bearer error="insufficient_scope", scope="<required>", resource="<mcp_resource_url>"`.
   - رفض أي طلب رمزي بدون المعلمة `code_verifier` أو `resource`.
الرفض الصارم (لا تقم أبدًا بإرسال أي من هذه الأشياء - ارفض الطلب وقم بتوثيق السبب):
- تخزين `client_secret` بنص عادي في متجر الحالة iii. يستخدم العملاء العامون `token_endpoint_auth_method: none`؛ يستخدم العملاء السريون `private_key_jwt`. لا توجد أسرار مشتركة بنص عادي في `state::*` أو في سجلات استجابة التسجيل.
- تخطي فحص `aud` على أداة التحقق. النائب الحائر هو السبب الكامل لـ RFC 8707 + RFC 9728.
- السماح بطلبات رمز التفويض الأقل PKCE. OAuth 2.1 يحظر ذلك؛ يجب على المدقق رفض أي تبادل `/token` الذي يفتقر سجل رمز التفويض المخزن إلى `code_challenge`.
- التخزين المؤقت JWKS بدون مهمة التحديث. إما أن يتم تشغيل سفن تشغيل cron، أو أن سطح المصادقة لا يتم نشره.
- الوثوق بمطالبة `iss` بدون قائمة السماح. يتيح أي مدقق يقبل رمزًا مميزًا من أي `iss` للمهاجم إنشاء موفر الهوية (IdP) الخاص به وتزوير الرموز المميزة.
- تخزين `registration_access_token` بنص عادي. التجزئة في الراحة. تتطلب نصًا واضحًا في كل تحديث.
الإخراج: خطة توصيل من صفحة واحدة مع مستند الموارد المحمية، واستدعاءات `registerTrigger` الثلاثة، واستدعاءات `registerFunction` الأربعة، وخطة مفتاح الحالة، وجدول تعيين النطاق، وقواعد رفض وقت التشغيل المشفرة. انتهى مع وجود فجوة حظر النشر الفردية التي من المرجح أن تظهر مقابل موفر الهوية الذي تم اختياره - عادةً ما يكون توفر DCR للمؤسسة SSO.