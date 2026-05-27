# MCP Auth in Production — DCR, JWKS Rotation, Audience-Pinned Tokens on iii Primitives

> الدرس 16 يوقف جهاز حالة OAuth 2.1 في الذاكرة. بحلول عام 2026، كل خادم MCP تشحنه إلى مؤسسة حقيقية يقع خلف مصادقة الإنتاج: تسجيل العميل الديناميكي (RFC 7591)، واكتشاف البيانات التعريفية لخادم التفويض (RFC 8414)، والتناوب JWKS الذي لا يكسر التحقق من صحة الرمز المميز في الساعة 3 صباحًا، والرموز المميزة المثبتة بالجمهور والتي ترفض إعادة استخدام النائب المرتبك. يربط هذا الدرس كل ذلك من خلال العناصر الأولية iii - `iii.registerTrigger` لـ HTTP وcron، `iii.registerFunction` لمنطق المصادقة، `state::set/get` للمفاتيح المخزنة مؤقتًا - بحيث يكون سطح المصادقة قابلاً للملاحظة وقابلاً لإعادة التشغيل وإعادة التشغيل مثل أي عبء عمل آخر في المحرك.

**النوع:** بناء
**اللغات:** Python (stdlib, iii primitives تم الاستهزاء بها في بيئة الدرس)
**المتطلبات الأساسية:** المرحلة 13 · 16 (جهاز حالة OAuth 2.1)، المرحلة 13 · 17 (البوابات)
**الوقت:** ~90 دقيقة

## Learning Objectives

- اكتشف خادم الترخيص من خلال RFC 8414 البيانات الوصفية وتحقق من العقد.
- تنفيذ RFC 7591 تسجيل العميل الديناميكي بحيث يتم تسجيل MCP العملاء دون تدخل المسؤول.
- تخزين مؤقت لمفاتيح JWKS وتدويرها باستخدام مشغل cron حتى ينجو التحقق من التوقيع من تمرير المفتاح.
- قم بتثبيت الرموز المميزة على مورد MCP واحد باستخدام RFC 8707 مؤشرات الموارد ورفض إعادة استخدام النائب المرتبك.
- قم بتوصيل كل نقطة نهاية ووظيفة خلفية كعناصر أولية iii - مشغلات HTTP، ومشغلات cron، والوظائف المسماة، وقراءات `state::*` — بحيث تؤدي إعادة التشغيل مرة واحدة إلى إعادة بناء سطح المصادقة.
- اقرأ مصفوفة قدرة IdP وارفض النشر عندما لا يتمكن IdP من تلبية ملف تعريف مصادقة MCP.

## The Problem

يقوم محاكي الدرس 16 بتشغيل OAuth 2.1 في الذاكرة. يحتوي الإنتاج على ثلاث فجوات تشغيلية لا يراها جهاز محاكاة الذاكرة فقط.

الفجوة الأولى هي التسجيل. تدير مؤسسة حقيقية مئات من MCP الخوادم وآلاف من MCP العملاء. لا يقوم المشغلون بتسجيل كل مستخدم للمؤشر يدويًا كعميل OAuth. RFC 7591 يتيح تسجيل العميل الديناميكي للعميل `POST /register` مقابل خادم الترخيص والحصول على `client_id` (واختياريًا `client_secret`) على الفور. ينشر الخادم `registration_endpoint` في بيانات التعريف RFC 8414؛ يكتشفه العميل دون تكوين خارج النطاق.

الفجوة الثانية هي دوران المفتاح. يعتمد التحقق من صحة JWT على مفاتيح التوقيع الخاصة بخادم التفويض، والتي تم نشرها كمجموعة مفاتيح ويب JSON (JWKS). يقوم خادم التفويض بتدوير هذه العناصر وفقًا لجدول زمني (غالبًا كل ساعة، وأحيانًا بشكل أسرع في ظل الاستجابة للحوادث). خادم MCP يجلب JWKS مرة واحدة عند التمهيد يتحقق من صحته بشكل جيد حتى نافذة التدوير - ثم يفشل كل طلب حتى إعادة التشغيل. أسلاك الإنتاج JWKS كقيمة مخبأة مع مهمة تحديث تقوم بالكتابة فوق ذاكرة التخزين المؤقت قبل انتهاء صلاحية المفاتيح السابقة، بالإضافة إلى الجلب الاحتياطي عند فقدان ذاكرة التخزين المؤقت للحالة التي يصل فيها رمز مميز موقّع بواسطة مفتاح أحدث من ذاكرة التخزين المؤقت.

والفجوة الثالثة هي ملزمة الجمهور. تم تقديم الدرس 16 RFC 8707 مؤشرات الموارد. في الإنتاج، يصبح هذا المؤشر بمثابة فحص مطالبة صعب لكل طلب. يقارن الخادم MCP `token.aud` بمورده الأساسي URL ويرفض عدم التطابق مع HTTP 401. هذا هو الدفاع الوحيد ضد خادم MCP المنبع (أو عميل ضار يحمل رمزًا مميزًا مخصصًا لخادم واحد) يعيد تشغيل هذا الرمز ضد خادم آخر في نفس شبكة الثقة.

يتعامل هذا الدرس مع كل واحدة من تلك الفجوات باعتبارها فجوة ثالثة بدائية. وثيقة البيانات التعريفية عبارة عن مشغل HTTP يُرجع مخرجات الوظيفة. التدوير JWKS هو مشغل cron الذي يستدعي `auth::rotate-jwks`، والذي يكتب إلى `state::set("auth/jwks/<issuer>",...)`. JWT التحقق من الصحة هو وظيفة يستدعيها الآخرون عبر `iii.trigger("auth::validate-jwt", token)`. الخادم MCP نفسه هو مجرد مشغل HTTP آخر يستدعي التحقق من الصحة قبل الإرسال. أعد تشغيل المحرك: تتم إعادة بناء سجل المشغل؛ الدولة باقية؛ سطح المصادقة يعمل بدون تسوية يدوية.

## The Concept

### RFC 8414 — OAuth Authorization Server Metadata

يصف المستند الموجود في `/.well-known/oauth-authorization-server` كل ما يحتاجه العميل:

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "registration_endpoint": "https://auth.example.com/register",
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

عميل حصل على MCP مورد URL اكتشاف السلاسل: `oauth-protected-resource` من RFC 9728 (مستند خادم المورد) يسمي المُصدر، ثم `oauth-authorization-server` (هذا RFC) يسمي كل نقطة نهاية. لا يقوم العميل مطلقًا بترميز التفويض URL.

العقد الذي تتحقق منه قبل الثقة في IdP لـ MCP:

- `code_challenge_methods_supported` يشمل `S256` (PKCE لكل RFC 7636).
- `grant_types_supported` تشمل `authorization_code` وترفض `password` و `implicit`.
- `registration_endpoint` موجود (RFC 7591 دعم).
- `response_types_supported` هو بالضبط `["code"]` لـ OAuth 2.1.

إذا كان أيًا منها مفقودًا، فسيرفض الخادم MCP النشر مقابل IdP هذا. بيان النشر خاطئ، وليس الكود.

### RFC 9728 (recap) — Protected Resource Metadata

تمت تغطية الدرس 16 RFC 9728. الدلتا في الإنتاج: هذا المستند هو المكان الوحيد الذي يبحث فيه العميل عن خوادم الترخيص الموثوق بها بواسطة *هذا* MCP الخادم. قد يقبل خادم MCP واحد الرموز المميزة من IdPs المتعددة (واحد للموظفين، وواحد للشركاء). RFC 9728 تعلن تلك المجموعة؛ RFC 8414 يوثق ما يدعمه كل IdP.

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### RFC 7591 — Dynamic Client Registration

بدون DCR، يحتاج كل عميل MCP (المؤشر، Claude Desktop، وكيل مخصص) إلى تبادل خارج النطاق مع مسؤول IdP. مع DCR، يقوم العميل بالنشر:

```json
POST /register
Content-Type: application/json

{
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

يستجيب الخادم بـ `client_id` و `registration_access_token` للتحديثات اللاحقة:

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`token_endpoint_auth_method: none` هو الإعداد الافتراضي الصحيح لعملاء MCP الذين يعملون على جهاز المستخدم. يحصلون على `client_id` فقط — لا `client_secret` للترشيح. PKCE يوفر إثبات الحيازة الذي يحتاجه العملاء العامون.

ثلاثة عيوب الإنتاج:

- يجب أن تكون نقطة نهاية التسجيل محددة حسب المصدر IP. وبدون ذلك، يقوم ممثل معادي بكتابة ملايين التسجيلات المزيفة واستنفاد مساحة الاسم `client_id`. iii make هذا أمر تافه: مشغل التسجيل HTTP يستدعي وظيفة `auth::rate-limit` قبل إرسالها إلى المسجل.
- `software_statement` (إقرار JWT موقع للعميل) مطلوب من قبل بعض موفري الهوية في المؤسسة. وهمية الدرس تتخطى ذلك؛ يمثل الإنتاج خطوة تحقق ترفض التسجيلات غير الموقعة من أي شيء آخر غير عناوين URI الخاصة بإعادة توجيه المضيف المحلي.
- يجب تخزين `registration_access_token` كعلامة تجزئة، وليس نصًا عاديًا. تعني سرقة هذا الرمز المميز أن المهاجم يمكنه إعادة كتابة عناوين URI لإعادة التوجيه الخاصة بالعميل.

### RFC 8707 (recap) — Resource Indicators

الدرس 16 حدد الشكل. قاعدة الإنتاج: يتضمن كل طلب رمز مميز `resource=<canonical-mcp-url>`، ويتحقق الخادم MCP من مطابقة `token.aud` للمورد الخاص به URL في كل مكالمة. إذا كان من الممكن الوصول إلى الخادم MCP عند `https://notes.example.com/mcp`, the canonical URL is `https://notes.example.com` — فسيتم استبعاد مكون المسار بحيث يستضيف خادم واحد مسارات متعددة ضمن جمهور واحد.

### RFC 7636 (recap) — PKCE

PKCE إلزامي في OAuth 2.1. يحمل تدفق رمز التفويض الخاص بالدرس دائمًا `code_challenge` و`code_verifier`. يرفض الخادم أي طلب رمز مميز بدون أداة تحقق أو باستخدام أداة تحقق لا تقوم بتجزئة الاختبار المخزن.

### MCP Spec 2025-11-25 Auth Profile

مواصفات MCP (25-11-2025) دقيقة بشأن ما يجب أن تفعله طبقة ترخيص الخادم MCP:

- نشر `/.well-known/oauth-protected-resource` (RFC 9728).
- قبول الرموز المميزة فقط عبر `Authorization: Bearer...`.
- التحقق من صحة `aud`، `iss`، `exp`، والنطاقات المطلوبة لكل طلب.
- أجب بـ `WWW-Authenticate` يحمل `Bearer error=...` لكل 401 و403، بما في ذلك المعلمات `scope=` و`resource=` حيثما ينطبق ذلك.
- رفض الرموز المميزة التي لا يتطابق رقم `aud` مع المورد الأساسي.
- رفض الرموز المميزة التي لا يوجد `iss` بها في قائمة `authorization_servers` لبيانات تعريف الموارد المحمية.

مسودة OAuth 2.1 هي الركيزة؛ RFC 8414/7591/8707/9728 + RFC 7636 هي السطح؛ MCP المواصفات هي الملف الشخصي.

### IdP capability matrix

لا يدعم كل IdP الملف الشخصي MCP الكامل. توثق المصفوفة أدناه بيانات القدرة الفعلية اعتبارًا من مواصفات 25-11-2025. إنها *بوابة نشر* وليست توصية.

| فئة موفِّر الهوية | RFC 8414 بيانات وصفية | RFC 7591 DCR | RFC 8707 مورد | RFC 7636 S256 PKCE | ملاحظات |
|---|---|---|---|---|---|
| استضافة ذاتية (Keycloak) | نعم | نعم | نعم (منذ 24.x) | نعم | مرجع IdP للملف الشخصي MCP في هذا الدرس؛ يدعم كل RFC نهاية إلى نهاية. |
| انتربرايز SSO (مايكروسوفت انترا ID) | نعم | نعم (المستويات المميزة) | نعم | نعم | DCR يختلف التوفر حسب فئة المستأجر؛ تحقق من المستأجر المستهدف قبل النشر. |
| إنتربرايز SSO (أوكتا) | نعم | نعم (Okta CIC / Auth0) | نعم | نعم | DCR متاح على Auth0 (الآن Okta CIC)؛ تتطلب مؤسسات Okta الكلاسيكية التسجيل المسبق للمسؤول. |
| معرفات تسجيل الدخول الاجتماعي (عامة) | يختلف | نادرا | نادرا | نعم | يعامل معظم مقدمي الهوية الاجتماعيين العملاء كشركاء ثابتين؛ لا تعتمد على DCR. استخدم كمصدر للهوية فقط، ضع طبقة خادم الترخيص الخاص بك MCP في الأعلى. |
| مخصص / محلي | يعتمد | يعتمد | يعتمد | يعتمد | إذا قمت بشحن الملف الشخصي الخاص بك، فقم بشحن الملف الشخصي بالكامل. يؤدي تخطي أي واحد من طلبات RFC الأربعة المذكورة أعلاه إلى انتهاك عقد المصادقة MCP. |

قاعدة رفض بيان النشر: إذا لم يُرجع IdP المختار `registration_endpoint` ولم يُدرج `S256` في `code_challenge_methods_supported`، فإن الخادم MCP يرفض البدء. لا يوجد وضع متدهور.

### JWKS rotation pattern with iii

وضع فشل الإنتاج عبارة عن ذاكرة تخزين مؤقت قديمة JWKS. قم بحلها باستخدام مشغل cron وذاكرة التخزين المؤقت `state::*`:

```python
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *", "name": "auth::jwks-refresh"},
    "auth::rotate-jwks",
)
```

كل ست ساعات، يقوم مشغل cron باستدعاء `auth::rotate-jwks`، الذي يجلب `<issuer>/.well-known/jwks.json` ويكتب إلى `state::set("auth/jwks/<issuer>", {keys, fetched_at})`. يقرأ المدقق من `state::get`. يقوم الرمز المميز الذي يكون `kid` مفقودًا من ذاكرة التخزين المؤقت بتشغيل استدعاء `auth::rotate-jwks` متزامن كإجراء احتياطي. يعالج هذا حالتين في وقت واحد: التدوير المجدول (كرون) ونوافذ تداخل المفاتيح (الرجوع المتزامن).

شكل الدولة:

```json
{
  "auth/jwks/https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

مفتاحين في وقت واحد هي الحالة المستقرة. يتم تدوير خوادم التفويض عن طريق تقديم المفتاح التالي (`k_2026_04`) قبل سحب المفتاح السابق (`k_2026_03`)، لذا تظل الرموز المميزة الصادرة بموجب المفتاح القديم صالحة حتى تنتهي صلاحيتها. ذاكرة التخزين المؤقت تحمل الاتحاد؛ يختار المدقق بواسطة `kid`.

### iii primitive wiring (the part this lesson is actually about)

خمسة أساسيات تشكل سطح المصادقة:

```python
# 1. RFC 8414 metadata document
iii.registerTrigger(
    "http",
    {"path": "/.well-known/oauth-authorization-server", "method": "GET"},
    "auth::serve-asm",
)

# 2. RFC 7591 dynamic client registration
iii.registerTrigger(
    "http",
    {"path": "/register", "method": "POST"},
    "auth::register-client",
)

# 3. JWT validation as a callable function (the resource server triggers it)
iii.registerFunction("auth::validate-jwt", validate_jwt_handler)

# 4. Step-up issuance for incremental scope (SEP-835 from L16)
iii.registerFunction("auth::issue-step-up", issue_step_up_handler)

# 5. Cron-driven JWKS rotation
iii.registerTrigger(
    "cron",
    {"schedule": "0 */6 * * *"},
    "auth::rotate-jwks",
)
iii.registerFunction("auth::rotate-jwks", rotate_jwks_handler)
```

الخادم MCP نفسه لا يستدعي التحقق من الصحة مباشرة. يفعل:

```python
result = iii.trigger("auth::validate-jwt", {"token": bearer_token, "resource": self.resource})
if not result["valid"]:
    return {"status": 401, "WWW-Authenticate": result["www_authenticate"]}
```

هذا المراوغة هو الرهان الثالث. غدًا، ستقوم بتبديل أداة التحقق من الصحة بمخرج متشعب يستشير اثنين من IdPs بالتوازي، أو تقوم بإضافة باعث الامتداد، أو تقوم بتخزين عمليات التحقق الإيجابية مؤقتًا. الخادم MCP لا يتغير.

### Confused-deputy walkthrough with audience binding

يتم تسجيل كل من الخادم A (`notes.example.com`) والخادم B (`tasks.example.com`) في نفس خادم التفويض. تم اختراق الخادم "أ". يأخذ المهاجم رمز ملاحظات المستخدم ويعيد تشغيله على الخادم B.

مدقق الخادم B:

1. فك التشفير JWT، جلب JWKS بواسطة `kid`، التحقق من التوقيع.
2. تحقق من `iss` مقابل بيانات تعريف الموارد المحمية `authorization_servers`. (تمرير - نفس IdP.)
3. حدد `aud == "https://tasks.example.com"`. (فشل - الرمز المميز `aud` هو `https://notes.example.com`.)
4. Return 401 with `WWW-المصادقة: خطأ الحامل = "invalid_token"، error_description = "جمهور غير متطابق"`.

مطالبة الجمهور هي الدفاع الوحيد ضد هذا الهجوم على طبقة البروتوكول. يعد تخطيه من أجل الأداء هو خطأ الإنتاج الأكثر شيوعًا؛ يجب أن يعمل المدقق عند كل طلب، وليس فقط عند بداية الجلسة.

### Failure modes

- **قديم JWKS.** يرفض المدقق الرموز الصالحة بعد تدوير المفتاح. الإصلاح هو نمط cron+fall-back أعلاه. لا تقم أبدًا بالتخزين المؤقت JWKS بدون مهمة التحديث.
- **مطالبة `aud` مفقودة.** يقوم بعض IdPs بحذف `aud` بشكل افتراضي ما لم يكن `resource` موجودًا في طلب الرمز المميز. يجب أن يرفض المدقق الرموز المميزة التي تحتوي على `aud` المفقودة، ولا يعامل الغياب كحرف بدل.
- **سباق ترقية النطاق.** يمكن لتدفقين متزامنين للترقية لنفس المستخدم أن ينجحا وينتجا رمزي وصول بنطاقات مختلفة. يجب على المدقق استخدام الرمز المميز المقدم في الطلب، وليس البحث عن "النطاق الحالي للمستخدم" - مما يؤدي إلى إنشاء نافذة TOCTOU.
- **سرقة رمز التسجيل المميز.** يتيح الرقم `registration_access_token` المسرب للمهاجم إعادة كتابة عناوين URI الخاصة بإعادة التوجيه. قم بتجزئة هذه الأشياء أثناء الراحة؛ مطالبة العميل بتقديم نص واضح في كل تحديث؛ تدور على الشبهة.
- **`iss` غير مثبت.** تتيح أداة التحقق التي تقبل أي `iss` للمهاجم إنشاء خادم التفويض الخاص به، وتسجيل عميل للجمهور المستهدف، وإصدار الرموز المميزة. القائمة `authorization_servers` لبيانات تعريف الموارد المحمية هي القائمة المسموح بها؛ فرضه.

## Use It

يمشي `code/main.py` على تدفق الإنتاج الكامل باستخدام stdlib Python وسجل `iii_mock` صغير يحاكي `iii.registerFunction`، `iii.registerTrigger`، `iii.trigger`، و`state::set/get`. التدفق:

1. ينشر خادم التفويض RFC 8414 بيانات وصفية على `/.well-known/oauth-authorization-server`.
2. MCP يستدعي العميل نقطة نهاية البيانات الوصفية، ويكتشف نقطة نهاية التسجيل.
3. MCP يرسل العميل إلى `/register` (RFC 7591) ويتلقى `client_id`.
4. MCP يقوم العميل بتشغيل تدفق كود التفويض المحمي PKCE (RFC 7636) مع المؤشر `resource` (RFC 8707).
5. يستدعي العميل MCP أداة على الخادم MCP باستخدام `Authorization: Bearer...`.
6. يقوم الخادم MCP بتشغيل `auth::validate-jwt`، والذي يقرأ JWKS من `state::get`.
7. يُطلق مشغل cron `auth::rotate-jwks`، ليحل محل JWKS في الحالة.
8. يتم التحقق من صحة المكالمة التالية مقابل المفاتيح الجديدة دون إعادة التشغيل.
9. محاولة نائب مشوش ضد مورد MCP مختلف تحصل على 401 مع عدم تطابق الجمهور.

يستخدم النموذج JWT هنا HS256 مع سر مشترك (وبالتالي يتم تشغيل الدرس على stdlib فقط). يستخدم الإنتاج RS256 أو EdDSA مع النمط JWKS أعلاه؛ منطق التحقق متطابق خلاف ذلك.

## Ship It

ينتج عن هذا الدرس `outputs/skill-mcp-auth-iii.md`. بالنظر إلى تكوين خادم MCP ومجموعة إمكانات IdP، تُصدر المهارة العناصر iii الأولية للتسجيل، وجدول التناوب JWKS، وتعيين النطاق، وقواعد الرفض التي سيتم تطبيقها عندما لا يدعم IdP ملف تعريف RFC الكامل.

## Exercises

1. قم بتشغيل `code/main.py`. تتبع التدفق المكون من 9 خطوات. لاحظ حيث يقوم `state::get` بإرجاع البيانات القديمة مباشرة قبل أن يقوم `auth::rotate-jwks` بالكتابة فوقها، وكيف يتم الآن التحقق من صحة الطلب التالي مقابل المفتاح الجديد.

2. قم بإضافة IdP جديد إلى قائمة بيانات تعريف الموارد المحمية `authorization_servers`. قم بإصدار رمز مميز موقع بواسطة IdP الجديد وتأكد من قبول المدقق له. قم بإصدار رمز مميز موقّع بواسطة IdP غير مدرج وتأكد من رفض أداة التحقق من خلال `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"`.

3. قم بتنفيذ `auth::rate-limit` كوظيفة iii واستدعائها من داخل مشغل التسجيل HTTP قبل تشغيل المسجل. استخدم دلو الرمز المميز لكل مصدر IP الموجود في `state::set("auth/ratelimit/<ip>",...)`.

4. اقرأ RFC 7591 وحدد حقلين لم يتحقق من صحة معالج الدرس `/register`. أضف التحقق من الصحة. (تلميح: مخطط `software_statement` و `redirect_uris` URI.)

5. اقرأ قسم التفويض MCP المواصفات 2025-11-25. ابحث عن المتطلب المعياري الوحيد في الرؤوس `WWW-Authenticate` التي لا يصدرها مدقق الدرس حاليًا. أضفه.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| ASM | "مستند بيانات تعريف OAuth" | RFC 8414 `/.well-known/oauth-authorization-server` JSON |
| DCR | "تسجيل عميل الخدمة الذاتية" | RFC 7591 `POST /register` تدفق |
| JWKS | "المفاتيح العامة للتحقق من صحة JWT" | JSON مجموعة مفاتيح الويب، تم جلبها من `jwks_uri`، مفهرسة بواسطة `kid` |
| مؤشر الموارد | "معلمة الجمهور" | RFC 8707 `resource` معلمة تثبيت الرمز المميز على خادم واحد |
| `aud` مطالبة | "الجمهور" | JWT المطالبة بمقارنة المدقق مع المصدر الأساسي URL |
| نائب حائر | "إعادة تشغيل الرمز المميز" | الهجوم حيث يتم تقديم الرمز المميز الصادر للخادم A إلى الخادم B |
| `iss` القائمة المسموح بها | "خوادم الترخيص الموثوقة" | المجموعة المسماة في بيانات تعريف الموارد المحمية `authorization_servers` |
| دوران المفتاح | "المتداول JWKS" | الاستبدال الدوري لمفاتيح التوقيع ذات النوافذ المتداخلة |
| العميل العام | "العميل الأصلي أو المتصفح" | عميل OAuth بدون الرقم `client_secret`؛ PKCE يعوض |
| `WWW-Authenticate` | "رأس الاستجابة 401/403" | يحمل التوجيهات `Bearer error=...` التي تدفع عملية استرداد العميل |

## Further Reading

- [MCP — Authorization spec (2025-11-25)](https://modelcontextprotocol.io/specification/draft/basic/authorization) — the MCP auth profile this lesson implements
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery contract
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol]( — https
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — إثبات ملكية العميل العام
- [RFC 8707 — مؤشرات الموارد لـ OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — تثبيت الجمهور
- [RFC 9728 — بيانات تعريف الموارد المحمية لـ OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc9728) — اكتشاف خادم الموارد
- [مسودة OAuth 2.1](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1) — ركيزة OAuth المدمجة
