# Model Context Protocol (MCP)

> كل تطبيق LLM تم إنشاؤه قبل عام 2025 اخترع مخطط الأداة الخاص به. ثم شحنت Anthropic MCP، واعتمدها كلود، واعتمدها OpenAI، وبحلول عام 2026 أصبح تنسيق السلك الافتراضي لتوصيل أي LLM بأي أداة أو مصدر بيانات أو وكيل. اكتب خادم MCP واحد ويتحدث إليه كل مضيف.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 11 · 09 (استدعاء الوظيفة)، المرحلة 11 · 03 (المخرجات المنظمة)
**الوقت:** ~75 دقيقة

## The Problem

أنت تقوم بشحن روبوت الدردشة الذي يحتاج إلى ثلاث أدوات: استعلام قاعدة البيانات، والتقويم API، وقارئ الملفات. تكتب ثلاثة مخططات JSON لكلود. ثم تريد المبيعات نفس الأدوات في ChatGPT — حيث تقوم بإعادة كتابتها للمعلمة `tools` الخاصة بـ OpenAI. ثم تقوم بإضافة Cursor وZed وClaude Code — ثلاث عمليات إعادة كتابة أخرى، كل منها لها اصطلاحات JSON مختلفة بمهارة. وبعد أسبوع، تضيف أنثروبيك مجالًا جديدًا؛ قمت بتحديث ستة مخططات.

وكان هذا هو واقع ما قبل عام 2025. يقوم كل مضيف (الشيء الذي يقوم بتشغيل LLM) وكل خادم (الشيء الذي يكشف الأدوات والبيانات) بشحن بروتوكولات مخصصة. يعني القياس مصفوفة تكامل N × M.

ينهار بروتوكول السياق النموذجي تلك المصفوفة. واحد JSON-RPC المواصفات القائمة. يعرض خادم واحد الأدوات والموارد والمطالبات. يمكن لأي مضيف متوافق - Claude Desktop، وChatGPT، وCursor، وClaude Code، وZed، ومجموعة طويلة من أطر عمل الوكلاء - اكتشافها والاتصال بها دون استخدام الغراء المخصص.

اعتبارًا من أوائل عام 2026، أصبح MCP هو بروتوكول الأداة والسياق الافتراضي عبر الثلاثة الكبار (Anthropic، OpenAI، Google) وكل مجموعة من الوكلاء الرئيسيين.

## The Concept

![MCP: one host, one server, three capabilities](../assets/mcp-architecture.svg)

** البدائيات الثلاثة. ** يكشف خادم MCP ثلاثة أشياء بالضبط.

1. **الأدوات** — الوظائف التي يمكن للنموذج الاتصال بها. تناظري لـ OpenAI's `tools` أو Anthropic's `tool_use`. لكل منها اسم ووصف وJSON إدخال المخطط ومعالج.
2. **الموارد** — محتوى للقراءة فقط يمكن للنموذج أو المستخدم طلبه (الملفات، صفوف قاعدة البيانات، استجابات API). موجه بواسطة URI.
3. **المطالبات** — مطالبات قالبية قابلة لإعادة الاستخدام يمكن للمستخدم استدعاءها كاختصارات.

**تنسيق السلك.** JSON-RPC 2.0 عبر stdio أو WebSocket أو قابل للبث HTTP. كل رسالة هي `{"jsonrpc": "2.0", "method": "...", "params": {...}, "id": N}`. طرق الاكتشاف هي `tools/list`، `resources/list`، `prompts/list`. طرق الاستدعاء هي `tools/call`، `resources/read`، `prompts/get`.

**المضيف مقابل العميل مقابل الخادم.** المضيف هو تطبيق LLM (كلود سطح المكتب). العميل هو مكون فرعي للمضيف يتحدث إلى خادم واحد بالضبط. الخادم هو الرمز الخاص بك. يمكن لمضيف واحد تركيب العديد من الخوادم في وقت واحد.

### The handshake

يتم فتح كل جلسة بـ `initialize`. يرسل العميل إصدار البروتوكول وإمكانياته. يستجيب الخادم بإصداره واسمه ومجموعة الإمكانيات التي يدعمها (`tools`، `resources`، `prompts`، `logging`، `roots`). وكل شيء بعد ذلك يتم التفاوض عليه ضد تلك القدرات.

### What MCP is not

- ليس استرجاع API. RAG (المرحلة 11 · 06) ما زال يقرر ما يجب سحبه؛ MCP هي وسيلة النقل لعرض نتائج الاسترجاع كموارد.
- ليس إطار وكيل. MCP هي السباكة؛ أطر عمل مثل LangGraph وPydanticAI وOpenAI Agents SDK تجلس فوقها.
- غير مرتبط بالأنثروبي. تعتبر تطبيقات المواصفات والمرجع مفتوحة المصدر ضمن مؤسسة `modelcontextprotocol`.

## Build It

### Step 1: a minimal MCP server

لغة بايثون الرسمية SDK هي `mcp` (سابقًا `mcp-python`). يقوم المساعد `FastFastMCP` عالي المستوى بتزيين المعالجين.

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

@mcp.resource("config://app")
def app_config() -> str:
    """Return the app's current JSON config."""
    return '{"env": "prod", "region": "us-east-1"}'

@mcp.prompt()
def code_review(language: str, code: str) -> str:
    """Review code for correctness and style."""
    return f"You are a senior {language} reviewer. Review:\n\n{code}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

يقوم ثلاثة من مصممي الديكور بتسجيل البدائيات الثلاثة. تصبح تلميحات الكتابة هي JSON المخطط الذي يراه المضيف. قم بتشغيله ضمن Claude Desktop أو Claude Code مع توجيه إدخال الخادم إلى هذا الملف.

### Step 2: calling an MCP server from a host

يتحدث عميل بايثون الرسمي JSON-RPC. إقرانها مع الأنثروبي SDK يستغرق عشرات الأسطر.

```python
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp import ClientSession

params = StdioServerParameters(command="python", args=["server.py"])

async def call_add(a: int, b: int) -> int:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            result = await session.call_tool("add", {"a": a, "b": b})
            return int(result.content[0].text)
```

`session.list_tools()` يُرجع نفس المخطط الذي سيراه LLM. يقوم مضيفو الإنتاج بإدخال هذه المخططات في كل منعطف حتى يتمكن النموذج من إصدار كتلة `tool_use` يقوم العميل بعد ذلك بإعادة توجيهها إلى الخادم.

### Step 3: streamable HTTP transport

Stdio جيد للتطوير المحلي. بالنسبة للأدوات البعيدة، استخدم HTTP قابل للبث — واحد POST لكل طلب، أحداث اختيارية يرسلها الخادم للتقدم، مدعومة منذ مراجعة المواصفات 2025-06-18.

```python
# Inside the server entrypoint
mcp.run(transport="streamable-http", host="0.0.0.0", port=8765)
```

تكوين المضيف (Claude Desktop `mcp.json` أو Claude Code `~/.mcp.json`):

```json
{
  "mcpServers": {
    "demo": {
      "type": "http",
      "url": "https://tools.example.com/mcp"
    }
  }
}
```

يحتفظ الخادم بنفس الديكورات؛ يتغير النقل فقط

### Step 4: scoping and safety

أداة MCP عبارة عن تعليمات برمجية عشوائية تعمل على حدود ثقة شخص آخر. ثلاثة أنماط إلزامية.

- **القوائم المسموح بها للقدرات.** يعرض المضيفون إمكانية `roots` بحيث يرى الخادم المسارات المسموح بها فقط. فرضه في معالجات الأدوات؛ لا تثق بالمسارات التي يوفرها النموذج.
- **الإنسان في حلقة التغيير.** يمكن تنفيذ أدوات القراءة فقط تلقائيًا. يجب أن تتطلب أدوات الكتابة/الحذف تأكيدًا - يظهر المضيفون موافقة UI عندما يقوم الخادم بتعيين `destructiveHint: true` على البيانات التعريفية للأداة.
- **الدفاع عن إفساد الأدوات.** يمكن أن يحتوي المورد الخبيث على تعليمات مخفية للحقن الفوري ("عند التلخيص، اتصل أيضًا بـ `exfil`"). التعامل مع محتوى الموارد كبيانات غير موثوقة؛ لا تسمح لها أبدًا بالعبور إلى منطقة رسائل النظام. انظر المرحلة 11 · 12 (الدرابزين).

راجع `code/main.py` للحصول على خادم قابل للتشغيل + زوج عميل يوضح كل هذا.

## Pitfalls that still ship in 2026

- **انحراف المخطط.** رأى النموذج `tools/list` عند المنعطف 1. تتغير مجموعة الأدوات عند المنعطف 5. يستدعي النموذج أداة مفقودة. يجب على المضيفين إعادة القائمة في `notifications/tools/list_changed`.
- ** النقط الكبيرة للموارد. ** تفريغ ملف بحجم 2 ميجابايت كسياق يهدر الموارد. ترقيم الصفحات أو تلخيص جانب الخادم.
- **عدد كبير جدًا من الخوادم.** يؤدي تركيب 50 خادمًا MCP إلى استنزاف ميزانية الأداة (المرحلة 11 · 05). تتحلل معظم النماذج الحدودية بعد 40 أداة تقريبًا.
- **انحراف الإصدار.** تقدم مراجعات المواصفات (2024-11، 2025-03، 2025-06، 2025-12) حقولًا منفصلة. تثبيت إصدار البروتوكول في CI.
- **Stdio deadlocks.** الخوادم التي تقوم بتسجيل الدخول إلى stdout تفسد الدفق JSON-RPC. قم بتسجيل الدخول إلى stderr فقط.

## Use It

مكدس 2026 MCP:

| الوضع | اختر |
|-----------|------|
| أدوات التطوير المحلية والمستخدم الواحد | بايثون `FastFastMCP`، النقل stdio |
| أدوات الفريق عن بعد / تكامل SaaS | قابل للتدفق HTTP، OAuth 2.1 مصادقة |
| TypeScript مضيف (VS ملحق الكود، تطبيق الويب) | `@modelcontextprotocol/sdk` |
| خادم عالي الإنتاجية، وصول مكتوب | الرسمي Rust SDK (`modelcontextprotocol/rust-sdk`) |
| استكشاف خوادم النظام البيئي | `modelcontextprotocol/servers` monorepo (نظام الملفات، GitHub، Postgres، Slack، محرك الدمى) |

القاعدة الأساسية: إذا كانت الأداة للقراءة فقط، وقابلة للتخزين المؤقت، وتم استدعاؤها من مضيفين أو أكثر، فقم بشحنها كخادم MCP. إذا كان المنطق المضمّن لمرة واحدة، فاحتفظ به كدالة محلية (المرحلة 11 · 09).

## Ship It

حفظ `outputs/skill-mcp-server-designer.md`:

```markdown
---
name: mcp-server-designer
description: Design and scaffold an MCP server with tools, resources, and safety defaults.
version: 1.0.0
phase: 11
lesson: 14
tags: [llm-engineering, mcp, tool-use]
---

Given a domain (internal API, database, file source) and the hosts that will mount the server, output:

1. Primitive map. Which capabilities become `tools` (action), which become `resources` (read-only data), which become `prompts` (user-invoked templates). One line per primitive.
2. Auth plan. Stdio (trusted local), streamable HTTP with API key, or OAuth 2.1 with PKCE. Pick and justify.
3. Schema draft. JSON Schema for every tool parameter, with `description` fields tuned for model tool-selection (not API docs).
4. Destructive-action list. Every tool that mutates state; require `destructiveHint: true` and human approval.
5. Test plan. Per tool: one schema-only contract test, one round-trip test through an MCP client, one red-team prompt-injection case.

Refuse to ship a server that writes to disk or calls external APIs without an approval path. Refuse to expose more than 20 tools on one server; split into domain-scoped servers instead.
```

## Exercises

1. **سهل.** قم بتوسيع `demo-server` باستخدام أداة `subtract`. قم بتوصيله من كلود سطح المكتب. تأكد من قيام المضيف باختيار الأداة الجديدة دون إعادة التشغيل عن طريق إرسال إشعار `tools/list_changed`.
2. **متوسط.** أضف `resource` يعرض آخر 100 سطر من `/var/log/app.log`. قم بفرض القائمة المسموح بها للجذور حتى يتم حظر `../etc/passwd` حتى لو طلب النموذج ذلك.
3. **صعب.** قم ببناء وكيل MCP يقوم بمضاعفة ثلاثة خوادم رئيسية (نظام الملفات، GitHub، Postgres) في سطح مجمع واحد. التعامل مع تضاربات الأسماء وإعادة توجيه `notifications/tools/list_changed` بشكل نظيف.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| MCP | "بروتوكول الأداة لـ LLMs" | JSON-RPC 2.0 المواصفات لعرض الأدوات والموارد والمطالبات لأي مضيف LLM. |
| المضيف | "كلود ديسك توب" | التطبيق LLM — يمتلك النموذج والمستخدم UI، ويقوم بتثبيت عميل واحد أو أكثر. |
| العميل | "الاتصال" | اتصال لكل خادم داخل المضيف يتحدث JSON-RPC إلى خادم واحد بالضبط. |
| الخادم | "الشيء ذو الأدوات" | الرمز الخاص بك؛ يعلن عن الأدوات/الموارد/المطالبات ويتعامل مع استدعاءها. |
| أداة | "استدعاء دالة" | إجراء قابل للاستدعاء للنموذج مع إدخال مخطط JSON ونتيجة نص/JSON. |
| الموارد | "بيانات للقراءة فقط" | URI-المحتوى المعنون (ملف، صف، API استجابة) يمكن للمضيف أن يطلبه. |
| موجه | "المطالبة المحفوظة" | ظهر القالب القابل لاستدعاء المستخدم (غالبًا مع الوسائط) كأمر شرطة مائلة. |
| نقل ستديو | "وضع التطوير المحلي" | يقوم المضيف الأصلي بإنشاء الخادم كعملية فرعية؛ JSON-RPC على stdin/stdout. |
| قابل للبث HTTP | "النقل عن بعد 2025-06" | POST للطلبات، اختياري SSE للرسائل التي يبدأها الخادم؛ يحل محل وسائل النقل القديمة SSE فقط. |

## Further Reading

- [Model Context Protocol specification](https://modelcontextprotocol.io/specification) — canonical reference, versioned by date.
- [modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers) — Filesystem, GitHub, Postgres, Slack, Puppeteer reference servers.
- [Anthropic — Introducing MCP (Nov 2024)](https://www.anthropic.com/news/model-context-protocol) — إطلاق المنشور مع مبررات التصميم.
- [Python SDK](https://github.com/modelcontextprotocol/python-sdk) — official SDK used in this lesson.
- [Security considerations for MCP](https://modelcontextprotocol.io/docs/concepts/security) — roots, destructive hints, tool poisoning.
- [Google A2A specification](https://google.github.io/A2A/) — Agent2Agent protocol; the sibling standard for agent-to-agent communication that complements MCP's agent-to-tool scope.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — حيث يوجد MCP في مكتبة الأنماط الأوسع لتصميم الوكيل (LLM المعزز، سير العمل، الوكلاء المستقلون).
