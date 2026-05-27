# Capstone 13 — MCP خادم مزود بالسجل والإدارة
> توقف بروتوكول السياق النموذجي عن كونه المستقبل وأصبح المواصفات الافتراضية لاستخدام الأداة في عام 2026. Anthropic وOpenAI وGoogle وكل عملاء IDE الرئيسيين الذين يشحنون MCP. نشرت Pinterest نظامها البيئي الداخلي لخوادم MCP. قام AAIF بإضفاء الطابع الرسمي على بيانات تعريف القدرة في `.well-known`. AWS ECS نشر النشر عديم الحالة المرجعي. وضع وكيل بلوك نفس البروتوكول داخل مساعد مستضاف. شكل الإنتاج لعام 2026 هو: نقل StreamableHTTP، ونطاقات OAuth 2.1، وبوابة السياسة OPA، ​​والسجل الذي يتيح لفرق النظام الأساسي اكتشاف الخوادم والتحقق من صحتها وتمكينها. بناء تلك النهاية إلى النهاية.
**النوع:** كابستون
**اللغات:** Python (الخادم، عبر FastMCP) أو TypeScript (@modelcontextprotocol/sdk)، Go (خدمة التسجيل)
**المتطلبات الأساسية:** المرحلة 11 (LLM الهندسة)، المرحلة 13 (الأدوات وMCP)، المرحلة 14 (الوكلاء)، المرحلة 17 (البنية التحتية)، المرحلة 18 (السلامة)
**مراحل التنفيذ:** P11 · P13 · P14 · P17 · P18
**الوقت:** 25 ساعة
## مشكلة
أصبح MCP لغة مشتركة لاستخدام الأدوات. Claude Code، وCursor 3، وAmp، وOpenCode، وGemini CLI، وكل وكيل مُدار يستهلك الآن خوادم MCP. لا تتمثل تحديات الإنتاج في تأليف الخوادم (FastMCP makes بهذه السهولة) ولكن نشرها على نطاق واسع مع متطلبات المؤسسة: نطاقات OAuth لكل مستأجر، وسياسة OPA بشأن الأدوات المدمرة، وقياس StreamableHTTP عديم الحالة، وسجل للاكتشاف، وسجلات التدقيق لكل استدعاء أداة. يقوم النظام البيئي MCP الداخلي لـ Pinterest ومواصفات التسجيل AAIF بتعيين شريط 2026.
ستنشئ خادم MCP يعرض 10 أدوات داخلية (Postgres للقراءة فقط، وقائمة S3، وJira، وLinear، وDatadog، وما إلى ذلك)، وسجل UI لاكتشاف النظام الأساسي، وبوابة موافقة بشرية للأدوات التدميرية. يوضح اختبار التحميل القياس الأفقي لـ StreamableHTTP. يفي مسار التدقيق بمراجعة أمان المؤسسة.
## مفهوم
MCP تتطلب مراجعة 2026 StreamableHTTP باعتباره وسيلة النقل الافتراضية. على عكس الشكل stdio-and-SSE السابق، يكون StreamableHTTP عديم الحالة افتراضيًا: تقبل نقطة نهاية HTTP واحدة طلبات JSON-RPC، وتدفق الاستجابات، وتدعم الاتصالات طويلة الأمد للإشعارات. يعني "عديم الحالة" أنه قابل للتطوير أفقيًا خلف موازن التحميل.
التفويض هو OAuth 2.1 مع نطاقات لكل أداة. يحمل الرمز نطاقات مثل `jira:read`، `s3:list`، `postgres:query:readonly`. يتحقق خادم MCP من النطاقات في وقت استدعاء الأداة، وليس فقط بداية الجلسة. بالنسبة للأدوات عالية المخاطر، يرفض الخادم أي استدعاء لم يتم رفع نطاقه إلى `approved:by:human` خلال آخر N دقيقة - ويأتي هذا الارتفاع من بطاقة مراجعة Slack.
التسجيل هو خدمة منفصلة. يعرض كل خادم MCP مستند `.well-known/mcp-capabilities` مع بيان الأداة الخاص به، والنقل URL، ومتطلبات المصادقة. يقوم باستطلاعات التسجيل والتحقق من الصحة والفهارس. تستخدم فرق النظام الأساسي السجل UI لمعرفة الأدوات المتاحة، والنطاقات التي يحتاجون إليها، والفرق التي تمتلكها.
## بنيان
```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## المكدس
- إطار عمل الخادم: FastMCP (Python) أو `@modelcontextprotocol/sdk` (TypeScript)
- النقل: StreamableHTTP عبر HTTPS (عديم الحالة)
- المصادقة: OAuth 2.1 مع هوية عبء العمل عبر SPIFFE / SPIRE
- السياسة: OPA / قواعد Rego لكل أداة؛ خدمة اتخاذ القرار السياسي لكل طلب
- التسجيل: مستضاف ذاتيًا، ويستهلك بيانات `.well-known/mcp-capabilities`
- موافقة الإنسان: رسالة تفاعلية Slack للأدوات التدميرية
- النشر: AWS ECS Fargate أو Fly.io، خادم واحد لكل مستأجر أو مشترك مع نطاق المستأجر
- التدقيق: مجموعة JSONL منظمة لكل مستأجر مع نسب كل مكالمة
## بنائها
1. **سطح الأداة.** كشف عن 10 أدوات داخلية: استعلام Postgres للقراءة فقط، S3 كائنات القائمة، بحث/جلب Jira، بحث/جلب خطي، استعلام متري Datadog، بحث PagerDuty عند الطلب، GitHub للقراءة فقط، بحث Notion، بحث Slack، قراءة Salesforce. تحتوي كل أداة على مخطط مكتوب وتسمية نطاق.
2. **خادم FastMCP.** قم بتثبيت الأدوات. قم بتكوين نقل StreamableHTTP. أضف برنامجًا وسيطًا لاستبطان رمز OAuth المميز وإنفاذ النطاق.
3. سياسة **OPA.** سياسة Rego لكل أداة: ما هي النطاقات التي تسمح بالاستدعاء، وما ينطبق عليه التنقيح PII، وما هي الحدود القصوى لحجم الحمولة المطبقة. يتم استدعاء خدمة القرار عند كل استدعاء للأداة.
4. **خدمة التسجيل.** خدمة Go منفصلة أو TS تستقصي `.well-known/mcp-capabilities` من الخوادم المسجلة، وتتحقق من صحتها باستخدام مخطط JSON، وتكشف عن قائمة / بحث / التحقق من صحة / تمكين-تعطيل UI.
5. **بيان القدرة.** يكشف كل خادم عن `.well-known/mcp-capabilities` مع: قائمة الأدوات، ومتطلبات المصادقة، والنقل URL، وفريق المالك، SLO.
6. **فصل الأدوات المدمر.** الأدوات التي تغير الحالة (إنشاء Jira، إنشاء Linear، كتابة Postgres) موجودة على خادم MCP ثاني مع تدفق مصادقة أكثر صرامة: يجب أن يكون للرموز المميزة نطاق `approved:by:human` مرتفع عبر بطاقة Slack في غضون 15 دقيقة.
7. **سجل التدقيق.** ألحق فقط JSONL لكل مستأجر: `{timestamp, user, tool, args_redacted, response_redacted, outcome}`. PII التنقيح عبر Presidio قبل الكتابة.
8. **اختبار التحميل.** 100 عميل متزامن على StreamableHTTP. إظهار القياس الأفقي عن طريق إضافة نسخة متماثلة ثانية؛ إظهار إعادة توزيع موازن التحميل دون التصاق الجلسة.
9. **اختبارات التوافق.** قم بتشغيل مجموعة التوافق الرسمية MCP على كلا الخادمين. اجتياز جميع الأقسام الإلزامية.
## استخدمه
```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## اشحنها
`outputs/skill-mcp-server.md` يصف التسليم. خادم MCP على مستوى الإنتاج + سجل + طبقة تدقيق للأدوات الداخلية مع نطاقات OAuth 2.1 وبوابات OPA.
| الوزن | المعيار | كيف يتم قياسه |
|:-:|---|---|
| 25 | مطابقة المواصفات | بيان قدرة StreamableHTTP + يجتاز اختبارات التوافق MCP |
| 20 | الأمن | تطبيق النطاق، OPA التغطية عبر كل أداة، النظافة السرية |
| 20 | إمكانية الملاحظة | سجل التدقيق لكل أداة مع تنقيح PII |
| 20 | مقياس | عرض توضيحي لاختبار الحمل على نطاق أفقي لـ 100 عميل |
| 15 | التسجيل UX | اكتشاف / التحقق من صحة / تمكين / تعطيل سير العمل |
| **100** | | |
## تمارين
1. إضافة أداة جديدة (بحث التقاء). قم بشحنه من خلال تدفق التحقق من صحة التسجيل دون لمس الخادم الأساسي.
2. اكتب سياسة OPA التي تعمل على تنقيح نتائج استعلام Postgres التي تحتوي على أعمدة مسماة `email`، أو `ssn`، أو `phone`. ممارسة مع استعلام التحقيق.
3. المعيار StreamableHTTP مقابل stdio على الكمون المحلي. تقرير لكل مكالمة ص50/ص95.
4. تنفيذ الحصة لكل مستأجر: الحد الأقصى لعدد N من المكالمات في الدقيقة لكل أداة لكل مستأجر. قم بالتنفيذ من خلال قاعدة OPA ثانية.
5. قم بتشغيل مجموعة التوافق MCP من [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) وأصلح كل فشل.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|---------------------------------------|
| StreamableHTTP | "2026 MCP النقل" | عديم الجنسية HTTP + البث؛ يستبدل SSE + stdio للخوادم المتصلة بالشبكة |
| بيان القدرة | "وثيقة مشهورة" | `.well-known/mcp-capabilities` مع قائمة الأدوات، والمصادقة، والنقل URL |
| OPA / ريجو | "محرك السياسة" | افتح وكيل السياسة لتخويل استدعاءات الأداة مقابل القواعد الخارجية |
| ارتفاع النطاق | "معتمد من الإنسان" | يتم منح النطاق قصير الأجل عبر موافقة Slack، وهو مطلوب للأدوات التدميرية |
| التسجيل | "اكتشاف الأداة" | الخدمة التي تقوم بفهرسة خوادم MCP من قدرتها تظهر |
| هوية عبء العمل | "SPIFFE / SPIRE" | هوية خدمة التشفير لإصدار رمز OAuth |
| جناح المطابقة | "اختبارات المواصفات" | بطارية اختبار MCP الرسمية لأداة StreamableHTTP + صحة البيان |
## مزيد من القراءة
- [Model Context Protocol 2026 Roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP، بيانات تعريف القدرة، التسجيل
- [AAIF MCP Registry spec](https://github.com/modelcontextprotocol/registry) — مواصفات التسجيل لعام 2026
- [AWS ECS reference deployment](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — نشر الإنتاج المرجعي
- [Pinterest internal MCP ecosystem](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — النشر الداخلي المرجعي
- [Block `goose` MCP usage](https://block.github.io/goose/) — نمط استهلاك الوكيل المرجعي
- [FastMCP](https://github.com/jlowin/fastmcp) — إطار عمل خادم بايثون
- [Open Policy Agent](https://www.openpolicyagent.org/) — مرجع محرك السياسة
- [SPIFFE / SPIRE](https://spiffe.io) — مرجع هوية عبء العمل