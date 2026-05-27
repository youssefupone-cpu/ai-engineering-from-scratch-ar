# MCP Apps — Interactive UI Resources via `ui://`

> تحدد مخرجات أداة النص فقط ما يمكن للوكلاء عرضه. MCP التطبيقات (SEP-1724، رسميًا في 26 يناير 2026) تسمح للأداة بإرجاع وضع الحماية التفاعلي HTML الذي تم تقديمه بشكل مضمّن في Claude Desktop وChatGPT وCursor وGoose وVS Code. لوحات المعلومات والنماذج والخرائط والمشاهد ثلاثية الأبعاد، كل ذلك من خلال امتداد واحد. يتناول هذا الدرس نظام الموارد `ui://` و`text/html;profile=mcp-app` MIME وبروتوكول postMessage لـ iframe-sandbox وسطح الأمان الذي يأتي مع السماح للخادم بعرض HTML.

**النوع:** بناء
** اللغات: ** بايثون (stdlib، UI باعث الموارد)، HTML (تطبيق نموذجي)
**المتطلبات:** المرحلة 13 · 07 (MCP خادم)، المرحلة 13 · 10 (الموارد)
**الوقت:** ~75 دقيقة

## Learning Objectives

- قم بإرجاع مورد `ui://` من استدعاء أداة وقم بتعيين MIME والبيانات التعريفية الصحيحة.
- قم بتعريف الأداة المرتبطة UI بـ `_meta.ui.resourceUri` و `_meta.ui.csp` و `_meta.ui.permissions`.
- تنفيذ رسالة postMessage الخاصة بوضع الحماية لـ iframe JSON-RPC للتواصل UI-إلى المضيف.
- تطبيق CSP وافتراضيات سياسة الأذونات التي تدافع ضد الهجمات التي نشأت UI.

## The Problem

يمكن لأداة `visualize_timeline` من حقبة 2025 إرجاع "إليك 14 ملاحظة منظمة ترتيبًا زمنيًا:...". تلك فقرة. يريد المستخدمون بالفعل الجدول الزمني التفاعلي. قبل تطبيقات MCP، كانت الخيارات هي: عنصر واجهة مستخدم خاص بالعميل APIs (قطع أثرية لكلود، OpenAI مخصص GPT HTML)، أو عدم وجود UI على الإطلاق.

MCP التطبيقات (SEP-1724، تم الشحن في 26 يناير 2026) توحيد العقد. تحتوي نتيجة الأداة على `resource` حيث URI هو `ui://...` وMIME هو `text/html;profile=mcp-app`. يعرضه المضيف في إطار iframe معزول بـ CSP محدود ولا يمكن الوصول إلى الشبكة ما لم يتم منحه صراحةً. يقوم UI داخل iframe بنشر الرسائل إلى المضيف عبر لهجة مشاركة صغيرة JSON-RPC.

يعرض كل عميل متوافق (Claude Desktop، ChatGPT، Goose، VS Code) نفس المورد `ui://` بنفس الطريقة. خادم واحد، حزمة HTML واحدة، عالمية UI.

## The Concept

### The `ui://` resource scheme

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

### Iframe sandbox

يعرض المضيف HTML داخل وضع الحماية `<iframe>` باستخدام:

- `sandbox="allow-scripts allow-same-origin"` (أو أكثر صرامة لكل إعلان خادم)
- تم إعلان CSP من قبل الخادم عبر رؤوس الاستجابة.
- لا توجد ملفات تعريف الارتباط، ولا يوجد تخزين محلي من أصل المضيف.
- يقتصر الوصول إلى الشبكة على `connectSrc` في CSP.

### postMessage protocol

يتواصل إطار iframe مع المضيف عبر `window.postMessage`. لهجة صغيرة JSON-RPC 2.0:

قم دائمًا بتثبيت `targetOrigin` في الأصل الدقيق للنظير، وعلى الجانب المتلقي، تحقق من صحة `event.origin` مقابل القائمة المسموح بها قبل معالجة أي حمولة. لا تستخدم أبدًا `"*"` لأي من جانبي هذه القناة — فالجسم يحمل استدعاءات الأدوات وقراءات الموارد.

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
- `host.readResource(uri)` — يقرأ المصدر MCP.
- `host.getPrompt(name, arguments)` — جلب قالب المطالبة.
- `host.close()` — يتجاهل UI.

لا تزال كل مكالمة تمر عبر بروتوكول MCP وترث أذونات الخادم.

### Permissions

تتطلب القائمة `_meta.ui.permissions` إمكانات إضافية:

- `camera` — الوصول إلى كاميرا المستخدم (المستخدمة لواجهات المستخدم الخاصة بمسح مستند ضوئيًا).
- `microphone` — الإدخال الصوتي.
- `geolocation` — الموقع.
- `network:*` — وصول أوسع إلى الشبكة مما يسمح به `connectSrc` وحده.

كل إذن عبارة عن مطالبة يراها المستخدم قبل عرض UI.

### Security risks

HTML في إطار iframe لا يزال HTML. سطح الهجوم الجديد:

- **الحقن الفوري عبر UI.** يمكن للخادم الضار UI إظهار نص يشبه رسالة النظام ويخدع المستخدم. يجب أن يميز عرض المضيف بشكل واضح الخادم UI عن المضيف UI.
- **الخروج عبر `connectSrc`.** إذا سمح CSP بـ `connect-src: *`، فيمكن لـ UI إرسال البيانات إلى أي مكان. يجب أن يكون الافتراضي صارما.
- **Clickjacking.** UI تراكب المضيف الكروم. يجب على المضيفين منع التلاعب بمؤشر z وفرض قواعد العتامة.
- **سرقة التركيز.** UI يأخذ تركيز لوحة المفاتيح ويلتقط الرسالة التالية. يجب على المضيفين اعتراض.

المرحلة 13 · 15 تغطي هذه الأمور بعمق كجزء من MCP الأمان؛ هذا الدرس يعرفهم.

### `ui/initialize` handshake

بعد تحميل iframe، يرسل `ui/initialize` عبر postMessage:

```json
{"jsonrpc": "2.0", "id": 0, "method": "ui/initialize",
 "params": {"theme": "dark", "locale": "en-US", "sessionId": "..."}}
```

يستجيب المضيف بالإمكانيات ورمز الجلسة. يستخدم UI رمز الجلسة في كل مكالمة مضيفة لاحقة.

### AppRenderer / AppFrame SDK primitives

تعرض التطبيقات الإضافية SDK اثنين من البدائيات الملائمة:

- `AppRenderer` (جانب الخادم) — يغلف مكون React / Vue / Solid ويصدر مورد `ui://` مع MIME الصحيح والبيانات الوصفية.
- `AppFrame` (جانب العميل) — يتلقى المورد، ويقوم بتثبيت iframe، ويتوسط postMessage.

يمكنك استخدامها أو لف HTML وJSON-RPC يدويًا.

### Ecosystem status

MCP تم شحن التطبيقات في 26 يناير 2026. دعم العملاء اعتبارًا من أبريل 2026:

- **Claude Desktop.** الدعم الكامل منذ يناير 2026.
- **ChatGPT.** الدعم الكامل عبر التطبيقات SDK (نفس بروتوكول التطبيقات MCP الأساسي).
- **المؤشر.** بيتا؛ تمكين عبر الإعدادات.
- **VS الكود.** تصميمات داخلية فقط.
- ** أوزة. ** الدعم الكامل.
- **Zed، Windsurf.** خريطة الطريق.

الخوادم قيد الإنتاج: لوحات المعلومات، وتصورات الخرائط، وجداول البيانات، وأدوات إنشاء المخططات، ومعاينات وضع الحماية IDE.

## Use It

`code/main.py` يوسع خادم الملاحظات بأداة `visualize_timeline` التي تُرجع مورد `ui://notes/timeline`، بالإضافة إلى معالج لـ `resources/read` على ذلك URI الذي يُرجع حزمة HTML صغيرة ولكن كاملة مع مخطط زمني SVG. HTML تم تصميمه بواسطة stdlib - لا يوجد نظام بناء. يتم رسم postMessage في JS التعليقات نظرًا لأن stdlib لا يمكنه تشغيل المتصفح.

ما الذي يجب النظر إليه:

- `_meta.ui` في استجابة الأداة تحمل أذونات الموارد Uri، CSP.
- يتم عرض HTML دون الوصول إلى الشبكة؛ جميع البيانات مضمنة.
- JS المكالمات `host.callTool` عبر `window.parent.postMessage` (موثقة ولكنها خاملة في هذا العرض التوضيحي stdlib).

## Ship It

ينتج عن هذا الدرس `outputs/skill-mcp-apps-spec.md`. نظرًا للأداة التي قد تستفيد من UI التفاعلية، تنتج المهارة عقد التطبيقات MCP الكامل: `ui://` URI، CSP، الأذونات، ونقاط دخول postMessage، وقائمة التحقق الأمنية.

## Exercises

1. قم بتشغيل `code/main.py` وتفقد HTML المنبعث. افتح HTML مباشرة في المتصفح؛ تحقق من العروض SVG. ثم قم برسم عقد postMessage الذي سيستخدمه UI للاتصال بـ `host.callTool("notes_update",...)`.

2. قم بتشديد CSP: قم بإزالة `'unsafe-inline'` واستخدم سياسة البرنامج النصي غير المستندة. ما هي التغييرات في رمز الجيل HTML؟

3. قم بإضافة مصدر UI ثاني `ui://notes/editor` مع نموذج لتحرير ملاحظة في مكانه. عندما يرسل المستخدم، يستدعي iframe `host.callTool("notes_update",...)`.

4. قم بتدقيق سطح هجوم UI. أين يمكن للخادم الضار إدخال محتوى؟ ما الذي يدافع عنه صندوق الحماية iframe وما الذي لا يدافع عنه؟

5. اقرأ مواصفات SEP-1724 وحدد قدرة واحدة في MCP التطبيقات SDK التي لا يستخدمها تطبيق اللعبة هذا. (تلميح: مزامنة الحالة على مستوى المكون.)

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| MCP تطبيقات | "مصادر UI تفاعلية" | SEP-1724 تم شحن التمديد بتاريخ 26-01-2026 |
| `ui://` | "مخطط التطبيق URI" | مخطط الموارد لحزم UI |
| `text/html;profile=mcp-app` | "الـ MIME" | نوع المحتوى لتطبيق MCP HTML |
| وضع الحماية لـ Iframe | "حاوية العرض" | وضع حماية المتصفح لـ UI مع CSP والأذونات |
| رسالة JSON-RPC | "UI-سلك المضيف" | صغيرة JSON-RPC-over-postMessage لهجة لمكالمات المضيف |
| `_meta.ui` | "أداة-UI ملزمة" | البيانات الوصفية التي تربط نتيجة الأداة بمورد UI |
| CSP | "سياسة أمن المحتوى" | يعلن عن المصادر المسموح بها للبرامج النصية والشبكات والأنماط |
| عارض التطبيقات | "الخادم SDK البدائي" | يحول مكون إطار العمل إلى مورد `ui://` |
| إطار التطبيق | "العميل SDK البدائي" | مساعد تثبيت Iframe الذي يتوسط postMessage |
| `ui/initialize` | "المصافحة" | أول مشاركةرسالة من UI للمضيف |

## Further Reading

- [MCP تطبيقات خارجية — GitHub](https://githubhub.com/modelcontextprotocol/ext-apps) — التنفيذ المرجعي و SDK
- [MCP مواصفات التطبيقات 26-01-2026](https://githubhub.com/modelcontextprotocol/ext-apps/blob/main/specification/2026-01-26/apps.mdx) — وثيقة المواصفات الرسمية
- [MCP — نظرة عامة على امتدادات التطبيقات](https://modelcontextprotocol.io/extensions/apps/overview) — وثائق عالية المستوى
- [مدونة MCP — MCP إطلاق التطبيقات](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/) — منشور الإطلاق في يناير 2026
- [MCP تطبيقات API مرجع](https://apps.extensions.modelcontextprotocol.io/api/) — مرجع SDK على نمط JSDoc
