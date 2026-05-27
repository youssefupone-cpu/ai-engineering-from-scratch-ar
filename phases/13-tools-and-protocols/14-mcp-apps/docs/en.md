# MCP التطبيقات — موارد UI التفاعلية عبر `ui://`
> تحدد مخرجات أداة النص فقط ما يمكن للوكلاء عرضه. تتيح تطبيقات MCP (SEP-1724، رسميًا في 26 يناير 2026) للأداة إرجاع HTML التفاعلية في وضع الحماية والتي تم عرضها مضمنة في Claude Desktop وChatGPT وCursor وGoose وVS Code. لوحات المعلومات والنماذج والخرائط والمشاهد ثلاثية الأبعاد، كل ذلك من خلال امتداد واحد. يتناول هذا الدرس نظام الموارد `ui://`، و`text/html;profile=mcp-app` MIME، وبروتوكول postMessage لـ iframe-sandbox، وسطح الأمان الذي يأتي مع السماح للخادم بعرض HTML.
**النوع:** بناء
**اللغات:** بايثون (stdlib، UI باعث الموارد)، HTML (تطبيق نموذجي)
**المتطلبات الأساسية:** المرحلة 13 · 07 (MCP الخادم)، المرحلة 13 · 10 (الموارد)
**الوقت:** ~75 دقيقة
## أهداف التعلم
- قم بإرجاع مورد `ui://` من استدعاء أداة وقم بتعيين MIME والبيانات التعريفية الصحيحة.
- أعلن عن أداة مرتبطة بـ UI بـ `_meta.ui.resourceUri`، `_meta.ui.csp`، و `_meta.ui.permissions`.
- قم بتنفيذ رسالة postMessage الخاصة بوضع الحماية لـ iframe JSON-RPC للاتصالات UI-to-host.
- تطبيق CSP والإعدادات الافتراضية لسياسة الأذونات التي تحمي من الهجمات التي تنشأ عن UI.
## المشكلة
يمكن لأداة `visualize_timeline` التي تعود إلى حقبة 2025 أن ترجع "إليك 14 ملاحظة منظمة ترتيبًا زمنيًا: ...". تلك فقرة. يريد المستخدمون بالفعل الجدول الزمني التفاعلي. قبل تطبيقات MCP، كانت الخيارات هي: واجهات برمجة تطبيقات عناصر واجهة المستخدم الخاصة بالعميل (عناصر Claude، OpenAI مخصص GPT HTML)، أو عدم وجود UI على الإطلاق.
MCP التطبيقات (SEP-1724، التي تم شحنها في 26 يناير 2026) تعمل على توحيد العقد. تحتوي نتيجة الأداة على `resource` الذي يكون URI هو `ui://...` وMIME هو `text/html;profile=mcp-app`. يعرضه المضيف في إطار iframe معزول بـ CSP محدود ولا يمكن الوصول إلى الشبكة ما لم يتم منحه صراحةً. يقوم UI الموجود داخل iframe بنشر الرسائل إلى المضيف عبر لهجة postMessage الصغيرة JSON-RPC.
يعرض كل عميل متوافق (Claude Desktop، ChatGPT، Goose، VS Code) نفس مورد `ui://` بنفس الطريقة. خادم واحد، حزمة HTML واحدة، عالمية UI.
##المفهوم
### نظام الموارد `ui://`
تقوم الأداة بإرجاع:
```json
{
  "content": [
    {"type": "text", "text": "Here is your notes timeline:"},
    {"type": "ui_resource", "uri": "ui://notes/timeline"}
  ],
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline",
      "csp": {
        "defaultSrc": "'self'",
        "scriptSrc": "'self' 'unsafe-inline'",
        "connectSrc": "'self'"
      },
      "permissions": []
    }
  }
}
```

ثم يتصل المضيف بـ `resources/read` على `ui://notes/timeline` URI ويعود:
```json
{
  "contents": [{
    "uri": "ui://notes/timeline",
    "mimeType": "text/html;profile=mcp-app",
    "text": "<!doctype html>..."
  }]
}
```

### صندوق الحماية Iframe
يعرض المضيف HTML داخل وضع الحماية `<iframe>` مع:
- `sandbox="allow-scripts allow-same-origin"` (أو أكثر صرامة لكل إعلان خادم)
- تم تطبيق CSP المعلن من قبل الخادم عبر رؤوس الاستجابة.
- لا توجد ملفات تعريف الارتباط، ولا يوجد تخزين محلي من أصل المضيف.
- يقتصر الوصول إلى الشبكة على `connectSrc` في CSP.
### بروتوكول ما بعد الرسالة
يتواصل إطار iframe مع المضيف عبر `window.postMessage`. لهجة صغيرة JSON-RPC 2.0:
قم دائمًا بتثبيت `targetOrigin` على المصدر الدقيق للنظير، وعلى الجانب المتلقي، تحقق من صحة `event.origin` مقابل القائمة المسموح بها قبل معالجة أي حمولة. لا تستخدم أبدًا `"*"` لأي من جانبي هذه القناة - فالجسم يحمل استدعاءات الأداة وقراءات الموارد.
```js
// iframe to host  (pin to host origin)
window.parent.postMessage({
  jsonrpc: "2.0",
  id: 1,
  method: "host.callTool",
  params: { name: "notes_update", arguments: { id: "note-14", title: "..." } }
}, "https://host.example.com");

// host to iframe  (pin to iframe origin)
iframe.contentWindow.postMessage({
  jsonrpc: "2.0",
  id: 1,
  result: { content: [...] }
}, "https://iframe.example.com");

// receiver on both sides
window.addEventListener("message", (event) => {
  if (event.origin !== "https://expected-peer.example.com") return;
  // safe to process event.data
});
```

الطرق المتاحة من جانب المضيف التي يمكن لـ UI الاتصال بها:
- `host.callTool(name, arguments)` — يستدعي أداة الخادم.
- `host.readResource(uri)` — يقرأ مورد MCP.
- `host.getPrompt(name, arguments)` — جلب قالب المطالبة.
- `host.close()` — يرفض UI.
لا تزال كل مكالمة تمر عبر بروتوكول MCP وترث أذونات الخادم.
### الأذونات
تتطلب قائمة `_meta.ui.permissions` إمكانيات إضافية:
- `camera` — الوصول إلى كاميرا المستخدم (المستخدمة لواجهات المستخدم الخاصة بمسح المستند ضوئيًا).
- `microphone` — الإدخال الصوتي.
- `geolocation` — الموقع.
- `network:*` — وصول إلى شبكة أوسع مما يسمح به `connectSrc` وحده.
كل إذن عبارة عن مطالبة يراها المستخدم قبل عرض UI.
### المخاطر الأمنية
HTML في إطار iframe لا يزال HTML. سطح الهجوم الجديد:
- **الحقن الفوري عبر UI.** يمكن للخادم الضار UI عرض نص يشبه رسالة النظام ويخدع المستخدم. يجب أن يميز عرض المضيف بشكل واضح بين الخادم UI والمضيف UI.
- **التسلل عبر `connectSrc`.** إذا كان CSP يسمح بـ `connect-src: *`، فيمكن لـ UI إرسال البيانات إلى أي مكان. يجب أن يكون الافتراضي صارما.
- **Clickjacking.** UI يتراكب مع الكروم المضيف. يجب على المضيفين منع التلاعب بمؤشر z وفرض قواعد العتامة.
- **سرقة التركيز.** UI يأخذ تركيز لوحة المفاتيح ويلتقط الرسالة التالية. يجب على المضيفين اعتراض.
تغطي المرحلة 13 · 15 هذه الأمور بعمق كجزء من MCP الأمان؛ هذا الدرس يعرفهم.
### `ui/initialize` المصافحة
بعد تحميل iframe، يرسل `ui/initialize` عبر postMessage:
```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

يستجيب المضيف بالإمكانيات ورمز الجلسة. يستخدم UI رمز الجلسة في كل مكالمة مضيفة لاحقة.
### AppRenderer / AppFrame SDK الأوليات
تكشف التطبيقات ext SDK عن اثنين من البدائيات الملائمة:
- `AppRenderer` (جانب الخادم) - يغلف مكون React / Vue / Solid ويصدر مورد `ui://` مع MIME والبيانات الوصفية الصحيحة.
- `AppFrame` (جانب العميل) - يتلقى المورد، ويقوم بتثبيت iframe، ويتوسط postMessage.
يمكنك استخدام هذه أو تمرير HTML وJSON-RPC يدويًا.
### حالة النظام البيئي
MCP تم شحن التطبيقات في 26 يناير 2026. دعم العملاء اعتبارًا من أبريل 2026:
- **Claude Desktop.** الدعم الكامل منذ يناير 2026.
- **ChatGPT.** الدعم الكامل عبر التطبيقات SDK (نفس بروتوكول التطبيقات MCP الأساسي).
- **المؤشر.** بيتا؛ تمكين عبر الإعدادات.
- **VS الكود.** الإصدارات الداخلية فقط.
- ** أوزة. ** الدعم الكامل.
- **Zed، Windsurf.** خريطة الطريق.
الخوادم قيد الإنتاج: لوحات المعلومات، وتصورات الخرائط، وجداول البيانات، وأدوات إنشاء المخططات، ومعاينات وضع الحماية IDE.
## استخدمه
يقوم `code/main.py` بتوسيع خادم الملاحظات باستخدام أداة `visualize_timeline` التي تُرجع مورد `ui://notes/timeline`، بالإضافة إلى معالج لـ `resources/read` على URI الذي يُرجع حزمة HTML صغيرة ولكنها كاملة مع مخطط زمني SVG. تم تصميم HTML وفقًا لنموذج stdlib — ولا يوجد نظام بناء. تم رسم postMessage في تعليقات JS نظرًا لأن stdlib لا يمكنه تشغيل المتصفح.
ما الذي يجب النظر إليه:
- `_meta.ui` في استجابة الأداة يحمل أذونات ResourcesUri، CSP.
- يتم عرض HTML بدون الوصول إلى الشبكة؛ جميع البيانات مضمنة.
- JS يستدعي `host.callTool` عبر `window.parent.postMessage` (موثق ولكنه خامل في هذا العرض التوضيحي stdlib).
## اشحنها
ينتج هذا الدرس `outputs/skill-mcp-apps-spec.md`. بالنظر إلى الأداة التي قد تستفيد من UI التفاعلي، تنتج المهارة عقد MCP Apps الكامل: `ui://` URI، CSP، والأذونات، ونقاط إدخال postMessage، وقائمة التحقق من الأمان.
## تمارين
1. قم بتشغيل `code/main.py` وافحص HTML المنبعث. افتح HTML مباشرة في المتصفح؛ تحقق من العروض SVG. ثم قم برسم عقد postMessage الذي سيستخدمه UI للاتصال بـ `host.callTool("notes_update", ...)`.
2. قم بتشديد CSP: قم بإزالة `'unsafe-inline'` واستخدم سياسة البرامج النصية غير القائمة. ما هي التغييرات في كود الجيل HTML؟
3. قم بإضافة مصدر UI ثاني `ui://notes/editor` مع نموذج لتحرير ملاحظة في مكانه. عندما يرسل المستخدم، يستدعي إطار iframe `host.callTool("notes_update", ...)`.
4. قم بمراجعة سطح الهجوم الخاص بـ UI. أين يمكن للخادم الضار إدخال محتوى؟ ما الذي يدافع عنه صندوق الحماية iframe وما الذي لا يدافع عنه؟
5. اقرأ مواصفات SEP-1724 وحدد قدرة واحدة في MCP Apps SDK التي لا يستخدمها تطبيق اللعبة هذا. (تلميح: مزامنة الحالة على مستوى المكون.)
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| MCP التطبيقات | "موارد UI التفاعلية" | SEP-1724 تم شحن الامتداد بتاريخ 26-01-2026 |
| `ui://` | "مخطط التطبيق URI" | مخطط الموارد لحزم UI |
| __الكود_1__ | "__المصطلح_5__" | نوع المحتوى لتطبيق MCP HTML |
| وضع الحماية لـ Iframe | "حاوية العرض" | وضع الحماية للمتصفح لـ UI مع CSP والأذونات |
| رسالة بريدية JSON-RPC | "UI-سلك للمضيف" | صغيرة JSON-RPC-over-postMessage لهجة المكالمات المضيفة |
| __الكود_2__ | "الأداة-UI ملزمة" | البيانات الوصفية التي تربط نتيجة الأداة بمورد UI |
| CSP | "سياسة أمن المحتوى" | يعلن عن المصادر المسموح بها للبرامج النصية والشبكات والأنماط |
| عارض التطبيقات | "الخادم SDK بدائي" | يحول مكون إطار العمل إلى مورد `ui://` |
| إطار التطبيق | "العميل SDK بدائي" | مساعد تثبيت Iframe الذي يتوسط postMessage |
| __الكود_4__ | "المصافحة" | الرسالة الأولى من UI إلى المضيف |
## مزيد من القراءة
- [MCP ext-apps — GitHub](https://github.com/modelcontextprotocol/ext-apps) — التنفيذ المرجعي وSDK
- [MCP Apps specification 2026-01-26](https://github.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — وثيقة المواصفات الرسمية
- [MCP — Apps extension overview](https://modelcontextprotocol.io/extensions/apps/overview) — وثائق رفيعة المستوى
- [MCP blog — MCP Apps launch](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — منشور الإطلاق لشهر يناير 2026
- [MCP Apps API reference](https://apps.extensions.modelcontextprotocol.io/api/) — مرجع بنمط JSDoc SDK