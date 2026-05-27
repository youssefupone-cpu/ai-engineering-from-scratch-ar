# Building an MCP Server — Python + TypeScript SDKs

> تظهر معظم البرامج التعليمية MCP فقط stdio hello-worlds. يعرض الخادم الحقيقي الأدوات بالإضافة إلى الموارد بالإضافة إلى المطالبات، ويتعامل مع التفاوض بشأن الإمكانيات، ويصدر أخطاء منظمة، ويعمل بنفس الطريقة عبر حزم SDK. يبني هذا الدرس خادم ملاحظات من البداية إلى النهاية: النقل stdlib stdio، الإرسال JSON-RPC، أوليات الخادم الثلاثة، ونمط الوظيفة النقية الذي يسقط إما في Python SDK's FastMCP أو TypeScript SDK عند التخرج.

**النوع:** بناء
** اللغات: ** بايثون (stdlib، stdio MCP الخادم)
**المتطلبات الأساسية:** المرحلة 13 · 06 (MCP الأساسيات)
**الوقت:** ~75 دقيقة

## Learning Objectives

- تنفيذ الطرق `initialize`، `tools/list`، `tools/call`، `resources/list`، `resources/read`، `prompts/list`، و `prompts/get`.
- اكتب حلقة إرسال تقرأ JSON-RPC رسائل من stdin وتكتب الردود على stdout.
- قم بإصدار استجابات للأخطاء المنظمة وفقًا لمواصفات JSON-RPC 2.0 ورموز MCP الإضافية.
- تخريج تطبيق stdlib إلى FastMCP (Python SDK) أو TypeScript SDK بدون إعادة كتابة منطق الأداة.

## The Problem

قبل أن تتمكن من استخدام النقل عن بعد (المرحلة 13 · 09) أو طبقة المصادقة (المرحلة 13 · 16)، تحتاج إلى خادم محلي نظيف. يعني المحلي stdio: يتم إنشاء الخادم بواسطة العميل كعملية فرعية، وتتدفق الرسائل عبر stdin/stdout مفصولة بسطر جديد.

تنص مواصفات 25-11-2025 على تشفير رسائل stdio ككائنات JSON مع فاصل `\n` صريح. لا SSE هنا؛ SSE كان الوضع البعيد القديم وستتم إزالته في منتصف عام 2026 (أوقفه خادم Atlassian's Rovo MCP في 30 يونيو 2026؛ وKeboola في 1 أبريل 2026). بالنسبة إلى stdio، كائن JSON واحد في كل سطر هو تنسيق السلك بالكامل.

يعد خادم الملاحظات شكلاً جيدًا لأنه يمارس جميع أساسيات الخادم الثلاثة. الأدوات تقوم بالطفرات (`notes_create`). تعرض الموارد البيانات (`notes://{id}`). يطالب قوالب الشحن (`review_note`). شكل هذا الدرس يعمم على أي مجال.

## The Concept

### Dispatch loop

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

- لا تطبع أي شيء على stdout ليس ظرفًا JSON-RPC. تذهب سجلات التصحيح إلى stderr.
- كل طلب MUST يجب أن يقابله رد يحمل نفس الرقم `id`.
- الإخطارات MUST NOT سيتم الرد عليها.

### Implementing `initialize`

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

### Implementing `tools/list` and `tools/call`

`tools/list` تُرجع `{tools: [...]}` مع كل إدخال يحتوي على `name`، `description`، `inputSchema`. `tools/call` تأخذ `{name, arguments}` وترجع andreturns`content:[blocks],isError:bool`.

تتم كتابة كتل المحتوى. الأكثر شيوعا:

```json
{"type": "text", "text": "Found 2 notes"}
{"type": "resource", "resource": {"uri": "notes://14", "text": "..."}}
{"type": "image", "data": "<base64>", "mimeType": "image/png"}
```

أخطاء الأداة تأتي في شكلين. الأخطاء على مستوى البروتوكول (طريقة غير معروفة، معلمات سيئة) هي JSON-RPC أخطاء. يتم إرجاع الأخطاء على مستوى الأداة (استدعاء صالح ولكن الأداة فاشلة) بالشكل `{content: [...], isError: true}`. يتيح ذلك للنموذج رؤية الفشل في سياقه.

### Implementing resources

الموارد للقراءة فقط حسب التصميم. `resources/list` يُرجع بيانًا؛ `resources/read` يقوم بإرجاع المحتوى. يمكن أن تكون عناوين URL `file://...`، `http://...`, or a custom scheme like `ملاحظات://`.

عندما تعرض البيانات كمورد بدلاً من أداة:

- النموذج لا "يستدعيه"؛ يمكن للعميل إدخاله في السياق بناءً على طلب المستخدم.
- تتيح الاشتراكات للخادم دفع التحديثات عندما يتغير المورد (المرحلة 13 · 10).
- المرحلة 13 · 14 توسع هذا بـ `ui://` للموارد التفاعلية.

### Implementing prompts

المطالبات هي قوالب ذات وسيطات مسماة. يعرضها المضيف كأوامر شرطة مائلة. قد تأخذ المطالبة `review_note` وسيطة `note_id` وتنتج قالب مطالبة متعدد الرسائل يقوم العميل بإدخاله إلى النموذج الخاص به.

### Stdio transport subtleties

- محدد بالخط الجديد JSON. لا يوجد إطار محدد للطول.
- لا المخزن المؤقت. `sys.stdout.flush()` بعد كل كتابة.
- العميل يتحكم في مدى الحياة. عندما يغلق stdin (EOF)، اخرج بشكل نظيف.
- لا تتعامل مع SIGPIPE بصمت؛ سجل والخروج.

### Annotations

يمكن لكل أداة أن تحمل `annotations` وصف خصائص السلامة:

- `readOnlyHint: true` — قراءة نقية، آمنة لإعادة المحاولة.
- `destructiveHint: true` — آثار جانبية لا رجعة فيها؛ يجب على العميل تأكيد.
- `idempotentHint: true` — نفس المدخلات تنتج نفس المخرجات.
- `openWorldHint: true` — يتفاعل مع الأنظمة الخارجية.

يستخدمها العميل لتحديد UX (مربعات حوار التأكيد، ومؤشرات الحالة) والتوجيه (المرحلة 13 · 17).

### Graduation path

يتكون خادم stdlib في `code/main.py` من حوالي 180 سطرًا. FastMCP (Python) ينهار نفس المنطق على نمط الديكور:

```python
from fastmcp import FastMCP
app = FastMCP("notes")

@app.tool()
def notes_search(query: str, limit: int = 10) -> list[dict]:
    ...
```

TypeScript SDK له شكل مكافئ. مسار التخرج متاح لك عندما تكون جاهزًا؛ المفاهيم (القدرات، والإرسال، وكتل المحتوى) هي نفسها.

## Use It

`code/main.py` هو خادم ملاحظات كامل MCP عبر stdio، stdlib فقط. يتعامل مع `initialize`، `tools/list`، `tools/call` لثلاث أدوات (`notes_list`، `notes_search`، `notes_create`)، `resources/list` و `resources/read` لكل ملاحظة، ومطالبة `review_note`. يمكنك قيادتها عن طريق pipالرسائل JSON-RPC:

```
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | python main.py
```

ما الذي يجب النظر إليه:

- المرسل هو `dict[str, Callable]` مرتبط باسم الطريقة.
- يقوم كل منفذ تنفيذ للأداة بإرجاع قائمة بكتل المحتوى، وليس سلسلة مجردة.
- `isError: true` يتم ضبطه عند رفع المنفذ.

## Ship It

ينتج عن هذا الدرس `outputs/skill-mcp-server-scaffolder.md`. بالنظر إلى المجال (الملاحظات، التذاكر، الملفات، قاعدة البيانات)، تدعم المهارة خادم MCP بالأدوات / الموارد / المطالبات المناسبة وتقسيم SDK مسار التخرج.

## Exercises

1. قم بتشغيل `code/main.py` وقيادته باستخدام رسائل JSON-RPC يدوية الصنع. قم بالتمرين `notes_create`، ثم `resources/read` لاسترداد الملاحظة الجديدة.

2. قم بإضافة أداة `notes_delete` مع `annotations: {destructiveHint: true}`. تأكد من أن العميل سيظهر مربع حوار تأكيد (وهذا يتطلب مضيفًا حقيقيًا؛ يعمل Claude Desktop).

3. قم بتنفيذ `resources/subscribe` بحيث يدفع الخادم `notifications/resources/updated` كلما تم تعديل الملاحظة. إضافة مهمة Keepalive.

4. قم بنقل الخادم إلى FastMCP. يجب أن يتقلص ملف Python إلى أقل من 80 سطرًا. يجب أن يكون سلوك السلك متطابقًا؛ تحقق باستخدام نفس أداة الاختبار JSON-RPC.

5. اقرأ قسم `server/tools` الخاص بالمواصفات وحدد حقلاً واحدًا لتعريف الأداة الذي لم يتم تنفيذه في خادم هذا الدرس. (تلميح: هناك العديد منها؛ اختر واحدًا وأضفه.)

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| MCP الخادم | "الشيء الذي يفضح الأدوات" | العملية التي تتحدث MCP JSON-RPC عبر stdio أو HTTP |
| نقل ستديو | "نموذج العملية التابعة" | يتم إنشاء الخادم بواسطة العميل؛ يتواصل عبر stdin/stdout |
| مرسل | "طريقة التوجيه" | خريطة JSON-RPC اسم الطريقة لوظيفة المعالج |
| كتلة المحتوى | "قطعة نتيجة الأداة" | العنصر المكتوب في المصفوفة `content` لاستجابة الأداة |
| `isError` | "فشل على مستوى الأداة" | إشارات فشل الأداة؛ يميز عن الخطأ JSON-RPC |
| الشروحات | "تلميحات السلامة" | أعلام القراءة فقط / المدمرة / العاطلة / العالم المفتوح |
| سريعMCP | "بايثون SDK" | إطار عمل عالي المستوى قائم على الديكور أعلى بروتوكول MCP |
| المصدر URI | "بيانات قابلة للعنونة" | `file://`، `db://`، أو مخطط مخصص لتحديد المورد |
| قالب موجه | "موجز أمر الشرطة المائلة" | القالب الذي يوفره الخادم مع فتحات الوسائط لواجهات المستخدم المضيفة |
| إعلان القدرة | "تبديل الميزة" | تم الإعلان عن العلامات لكل بدائية في `initialize` |

## Further Reading

- [بروتوكول سياق النموذج — Python SDK](https://githubhub.com/modelcontextprotocol/python-sdk) — تنفيذ Python المرجعي
- [بروتوكول سياق النموذج — TypeScript SDK](https://githubhub.com/modelcontextprotocol/typescript-sdk) — التنفيذ المتوازي TS
- [FastMCP — إطار عمل الخادم](https://gofastmcp.com/) — Python على طراز الديكور API لخوادم MCP
- [MCP — دليل خادم البدء السريع](https://modelcontextprotocol.io/quickstart/server) — برنامج تعليمي شامل باستخدام إما SDK
- [MCP — مواصفات أدوات الخادم](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) — مرجع كامل للأدوات/* الرسائل
