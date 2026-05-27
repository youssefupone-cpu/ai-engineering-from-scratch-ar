# إنشاء خادم MCP — Python + TypeScript SDKs
> تعرض معظم البرامج التعليمية الخاصة بـ MCP فقط عوالم hello-worlds الخاصة بـ stdio. يعرض الخادم الحقيقي الأدوات بالإضافة إلى الموارد بالإضافة إلى المطالبات، ويتعامل مع التفاوض بشأن الإمكانيات، ويصدر أخطاء منظمة، ويعمل بنفس الطريقة عبر حزم SDK. يبني هذا الدرس خادم ملاحظات من البداية إلى النهاية: النقل stdlib stdio، JSON-RPC الإرسال، أوليات الخادم الثلاثة، ونمط الوظيفة النقية الذي يقع إما في FastMCP الخاص بـ Python SDK أو TypeScript SDK عند التخرج.
**النوع:** بناء
** اللغات: ** بايثون (خادم stdlib، stdio MCP)
**المتطلبات الأساسية:** المرحلة 13 · 06 (MCP الأساسيات)
**الوقت:** ~75 دقيقة
## أهداف التعلم
- تنفيذ الأساليب `initialize`، `tools/list`، `tools/call`، `resources/list`، `resources/read`، `prompts/list`، و`prompts/get`.
- اكتب حلقة إرسال تقرأ رسائل JSON-RPC من stdin وتكتب الردود على stdout.
- إرسال استجابات للأخطاء المنظمة وفقًا لمواصفات JSON-RPC 2.0 والرموز الإضافية لـ MCP.
- قم بتخريج تطبيق stdlib إلى FastMCP (Python SDK) أو TypeScript SDK بدون إعادة كتابة منطق الأداة.
## المشكلة
قبل أن تتمكن من استخدام النقل عن بعد (المرحلة 13 · 09) أو طبقة المصادقة (المرحلة 13 · 16)، تحتاج إلى خادم محلي نظيف. يعني المحلي stdio: يتم إنشاء الخادم بواسطة العميل كعملية فرعية، وتتدفق الرسائل عبر stdin/stdout مفصولة بسطر جديد.
تنص مواصفات 25-11-2025 على أن رسائل stdio يتم ترميزها ككائنات JSON مع فاصل `\n` صريح. لا يوجد SSE هنا؛ كان SSE هو الوضع البعيد القديم وستتم إزالته في منتصف عام 2026 (أوقفه خادم Atlassian's Rovo MCP في 30 يونيو 2026؛ وKeboola في 1 أبريل 2026). بالنسبة إلى stdio، فإن كائن JSON واحد في كل سطر هو تنسيق السلك بالكامل.
يعد خادم الملاحظات شكلاً جيدًا لأنه يمارس جميع أساسيات الخادم الثلاثة. الأدوات تقوم بالطفرات (`notes_create`). تعرض الموارد البيانات (`notes://{id}`). يطالب قوالب السفينة (`review_note`). شكل هذا الدرس يعمم على أي مجال.
##المفهوم
### حلقة الإرسال
```
loop:
  line = stdin.readline()
  msg = json.loads(line)
  if has id:
    handle request -> write response
  else:
    handle notification -> no response
```

ثلاث قواعد:
- لا تطبع أي شيء على stdout ليس مظروفًا JSON-RPC. تذهب سجلات التصحيح إلى stderr.
- تتم مطابقة كل طلب MUST برد يحمل نفس `id`.
- سيتم الرد على الإخطارات MUST NOT.
### تنفيذ `initialize`
```python
def initialize(params):
    return {
        "protocolVersion": "2025-11-25",
        "capabilities": {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": False},
            "prompts": {"listChanged": False},
        },
        "serverInfo": {"name": "notes", "version": "1.0.0"},
    }
```

أعلن فقط ما تدعمه. يعتمد العميل على القدرة المعينة لميزات البوابة.
### تنفيذ `tools/list` و`tools/call`
`tools/list` يُرجع `{tools: [...]}` مع كل إدخال يحتوي على `name`، `description`، `inputSchema`. `tools/call` يأخذ `{name, arguments}` ويعيد `{content: [blocks], isError: bool}`.
تتم كتابة كتل المحتوى. الأكثر شيوعا:
```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

أخطاء الأداة تأتي في شكلين. الأخطاء على مستوى البروتوكول (أسلوب غير معروف، معلمات سيئة) هي أخطاء JSON-RPC. يتم إرجاع الأخطاء على مستوى الأداة (استدعاء صالح ولكن الأداة فاشلة) بالشكل `{content: [...], isError: true}`. يتيح ذلك للنموذج رؤية الفشل في سياقه.
### تنفيذ الموارد
الموارد للقراءة فقط حسب التصميم. `resources/list` يُرجع بيانًا؛ `resources/read` يقوم بإرجاع المحتوى. يمكن أن تكون عناوين URI `file://...`، أو `http://...`، أو نظامًا مخصصًا مثل `notes://`.
عندما تعرض البيانات كمورد بدلاً من أداة:
- النموذج لا "يستدعيه"؛ يمكن للعميل إدخاله في السياق بناءً على طلب المستخدم.
- تتيح الاشتراكات للخادم دفع التحديثات عندما يتغير المورد (المرحلة 13 · 10).
- المرحلة 13 · 14 توسع هذا باستخدام `ui://` للموارد التفاعلية.
### تنفيذ المطالبات
Prompts are templates with named arguments. The host surfaces them as slash-commands. قد تأخذ المطالبة `review_note` الوسيطة `note_id` وتنتج قالب مطالبة متعدد الرسائل يغذيه العميل إلى النموذج الخاص به.
### خفايا النقل Stdio
- محدد بسطر جديد JSON. لا يوجد إطار محدد للطول.
- لا المخزن المؤقت. `sys.stdout.flush()` بعد كل كتابة.
- العميل يتحكم في مدى الحياة. عندما يغلق stdin (EOF)، اخرج بشكل نظيف.
- لا تتعامل مع SIGPIPE بصمت؛ سجل والخروج.
### التعليقات التوضيحية
يمكن لكل أداة أن تحمل `annotations` الذي يصف خصائص السلامة:
- `readOnlyHint: true` — قراءة نقية، آمنة لإعادة المحاولة.
- `destructiveHint: true` — آثار جانبية لا رجعة فيها؛ يجب على العميل تأكيد.
- `idempotentHint: true` — نفس المدخلات تنتج نفس المخرجات.
- `openWorldHint: true` — يتفاعل مع الأنظمة الخارجية.
يستخدم العميل هذه لتحديد UX (مربعات حوار التأكيد، ومؤشرات الحالة) والتوجيه (المرحلة 13 · 17).
### مسار التخرج
يبلغ طول خادم stdlib في `code/main.py` حوالي 180 سطرًا. يقوم FastMCP (Python) بطي نفس المنطق على نمط الديكور:
```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK له شكل مكافئ. مسار التخرج متاح لك عندما تكون جاهزًا؛ المفاهيم (القدرات، والإرسال، وكتل المحتوى) هي نفسها.
## استخدمه
`code/main.py` عبارة عن خادم ملاحظات كامل MCP عبر stdio وstdlib فقط. يتعامل مع `initialize`، `tools/list`، `tools/call` لثلاث أدوات (`notes_list`، `notes_search`، `notes_create`)، `resources/list` و`resources/read` لكل ملاحظة، وموجه `review_note`. يمكنك قيادتها عن طريق piping JSON-RPC الرسائل:
```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

ما الذي يجب النظر إليه:
- المرسل هو `dict[str, Callable]` مرتبط باسم الطريقة.
- يقوم كل منفذ تنفيذ للأداة بإرجاع قائمة بكتل المحتوى، وليس سلسلة مجردة.
- `isError: true` يتم ضبطه عند رفع المنفذ.
## اشحنها
ينتج هذا الدرس `outputs/skill-mcp-server-scaffolder.md`. بالنظر إلى المجال (الملاحظات، التذاكر، الملفات، قاعدة البيانات)، تدعم المهارة خادم MCP بالأدوات / الموارد / المطالبات المناسبة وتقسيم SDK مسار التخرج.
## تمارين
1. قم بتشغيل `code/main.py` وقيادته باستخدام رسائل JSON-RPC المضمنة يدويًا. تمرين `notes_create`، ثم `resources/read` لاسترداد الملاحظة الجديدة.
2. أضف أداة `notes_delete` مع `annotations: {destructiveHint: true}`. تأكد من أن العميل سيظهر مربع حوار تأكيد (وهذا يتطلب مضيفًا حقيقيًا؛ يعمل Claude Desktop).
3. قم بتنفيذ `resources/subscribe` بحيث يدفع الخادم `notifications/resources/updated` كلما تم تعديل الملاحظة. إضافة مهمة Keepalive.
4. قم بتوصيل الخادم إلى FastMCP. يجب أن يتقلص ملف Python إلى أقل من 80 سطرًا. يجب أن يكون سلوك السلك متطابقًا؛ تحقق باستخدام نفس أداة الاختبار JSON-RPC.
5. اقرأ قسم `server/tools` الخاص بالمواصفات وحدد حقلاً واحدًا لتعريف الأداة الذي لم يتم تنفيذه في خادم هذا الدرس. (تلميح: هناك العديد منها؛ اختر واحدًا وأضفه.)
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| MCP الخادم | "الشيء الذي يفضح الأدوات" | العملية التي تتحدث MCP JSON-RPC عبر stdio أو HTTP |
| نقل ستديو | "نموذج العملية التابعة" | يتم إنشاء الخادم بواسطة العميل؛ يتواصل عبر stdin/stdout |
| مرسل | "طريقة التوجيه" | تعيين اسم الأسلوب JSON-RPC لوظيفة المعالج |
| كتلة المحتوى | "قطعة نتيجة الأداة" | العنصر المكتوب في مصفوفة `content` لاستجابة الأداة |
| __الكود_1__ | "فشل على مستوى الأداة" | إشارات فشل الأداة؛ يميز عن الخطأ JSON-RPC |
| الشروحات | "تلميحات السلامة" | أعلام القراءة فقط / المدمرة / العاطلة / العالم المفتوح |
| فاست إم سي بي | "بايثون SDK" | إطار عمل عالي المستوى قائم على الديكور أعلى بروتوكول MCP |
| المصدر URI | "بيانات قابلة للعنونة" | `file://`، `db://`، أو مخطط مخصص يحدد المورد |
| قالب موجه | "موجز أمر الشرطة المائلة" | القالب الذي يوفره الخادم مع فتحات الوسائط لواجهات المستخدم المضيفة |
| إعلان القدرة | "تبديل الميزة" | تم الإعلان عن العلامات لكل بدائية في `initialize` |
## مزيد من القراءة
- [Model Context Protocol — Python SDK](https://github.com/modelcontextprotocol/python-sdk) — تطبيق بايثون المرجعي
- [Model Context Protocol — TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk) — التنفيذ الموازي TS
- [FastMCP — server framework](https://gofastmcp.com/) — لغة Python ذات النمط المزخرف API لخوادم MCP
- [MCP — Quickstart server guide](https://modelcontextprotocol.io/quickstart/server) — برنامج تعليمي شامل باستخدام إما SDK
- [MCP — Server tools spec](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — مرجع كامل للأدوات/* الرسائل