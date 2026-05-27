# OpenTelemetry GenAI Semantic Conventions

> يحدد GenAI SIG الخاص بـ OpenTelemetry (تم إطلاقه في أبريل 2024) المخطط القياسي لقياس الوكيل عن بعد. تتقارب أسماء النطاقات والسمات وقواعد التقاط المحتوى عبر البائعين، لذا فإن آثار الوكيل تعني نفس الشيء في Datadog وGrafana وJaeger وHoneycomb.

** النوع: ** تعلم + بناء
** اللغات: ** بايثون (stdlib)
**المتطلبات الأساسية:** المرحلة 14 · 13 (LangGraph)، المرحلة 14 · 24 (منصات المراقبة)
**الوقت:** ~60 دقيقة

## Learning Objectives

- قم بتسمية فئات GenAI الممتدة: النموذج/العميل، الوكيل، الأداة.
- التمييز بين `invoke_agent` CLIENT مقابل المسافات الداخلية ومتى ينطبق كل منهما.
- قائمة سمات GenAI ذات المستوى الأعلى: اسم الموفر، نموذج الطلب، مصدر البيانات ID.
- شرح عقد التقاط المحتوى: الاشتراك، `OTEL_SEMCONV_STABILITY_OPT_IN`، توصية المرجع الخارجي.

## The Problem

يخترع كل بائع أسماء النطاقات الخاصة به. تنتهي فرق العمليات ببناء لوحات معلومات لكل إطار عمل. يعمل GenAI SIG الخاص بـ OpenTelemetry على إصلاح هذه المشكلة عن طريق تحديد معيار واحد يستهدفه النظام البيئي بأكمله.

## The Concept

### Span categories

1. **امتدادات النموذج/العميل.** تغطية مكالمات LLM الأولية. ينبعث من مزود SDKs (Anthropic، OpenAI، Bedrock) ومحولات نموذج الإطار.
2. **امتدادات الوكيل.** `create_agent` (عند إنشاء الوكيل) و `invoke_agent` (عند تشغيله).
3. **امتدادات الأداة.** واحد لكل استدعاء للأداة؛ متصلة بمدى الوكيل عن طريق العلاقة بين الوالدين والطفل.

### Agent span naming

- اسم النطاق: `invoke_agent {gen_ai.agent.name}` إذا تم تسميته؛ الرجوع إلى `invoke_agent`.
- نوع الامتداد: - **CLIENT** — لخدمات الوكلاء عن بعد (OpenAI المساعدون API، وكلاء Bedrock). - **داخلي** — لأطر عمل الوكلاء قيد التشغيل (LangChain، وCrewAI، وReAct المحلي).

### Key attributes

- `gen_ai.provider.name` — `anthropic`، `openai`، `aws.bedrock`، `google.vertex`.
- `gen_ai.request.model` — الموديل ID.
- `gen_ai.response.model` — النموذج الذي تم حله (قد يختلف عن الطلب بسبب التوجيه).
- `gen_ai.agent.name` — معرف الوكيل.
- `gen_ai.operation.name` — agentidentifier، ، ، .
- `gen_ai.data_source.id` — لـ RAG: أي مجموعة أو متجر تمت استشارته.

توجد اصطلاحات خاصة بالتكنولوجيا لـ Anthropic، Azure AI الاستدلال، AWS Bedrock، OpenAI.

### Content capture

القاعدة الافتراضية: الأجهزة SHOULD NOT تلتقط المدخلات/المخرجات بشكل افتراضي. يتم الاشتراك في الالتقاط عبر:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

نمط الإنتاج الموصى به: تخزين المحتوى خارجيًا (S3، مخزن السجل الخاص بك)، تسجيل المراجع على امتدادات (معرفات المؤشر، وليس النثر). هذا هو الدرس 27 من الدفاع عن تسميم المحتوى المرتبط بقابلية الملاحظة.

### Stability

معظم الاتفاقيات تجريبية اعتبارًا من مارس 2026. يمكنك الاشتراك في المعاينة الثابتة باستخدام:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

يقوم Datadog v1.37+ بتعيين سمات GenAI أصلاً في مخطط إمكانية الملاحظة LLM. تدعم الواجهات الخلفية الأخرى (Grafana، وHoneycomb، وJaeger) السمات الأولية.

### Where this pattern goes wrong

- **التقاط المطالبات الكاملة على فترات.** PII، الأسرار، بيانات العملاء في آثار يمكن للعمليات قراءتها. تخزين خارجيا.
- **لا `gen_ai.provider.name`.** تنقطع لوحات المعلومات متعددة الموفرين عند فقدان الإسناد.
- **امتدادات بدون روابط أصلية.** امتدادات أداة معزولة. دائما نشر السياق.
- **عدم تعيين خيار الاستقرار.** قد تتم إعادة تسمية السمات الخاصة بك عند ترقية الواجهة الخلفية.

## Build It

`code/main.py` يطبق باعث stdlib Span الذي يطابق اصطلاحات GenAI:

- `Span` مع مخطط سمة GenAI.
- `Tracer` مع `start_span`، سياقات متداخلة.
- تشغيل وكيل مكتوب يصدر: `create_agent`، `invoke_agent` (داخلي)، يمتد لكل أداة، `chat` يمتد لـ LLM من المكالمات.
- وضع التقاط المحتوى الذي يخزن المطالبات خارجيًا ويسجل المعرفات على امتدادات.

تشغيله:

```
python3 code/main.py
```

الإخراج: شجرة امتداد تحتوي على جميع سمات GenAI المطلوبة، و"متجر خارجي" يعرض مراجع محتوى الاشتراك.

## Use It

- **Datadog LLM إمكانية الملاحظة** (الإصدار 1.37+) يعين السمات محليًا.
- **Langfuse / Phoenix / Opik** (الدرس 24) — التجهيز التلقائي للنظام البيئي.
- **Jaeger / Honeycomb / Grafana Tempo** — آثار OTel الخام؛ بناء لوحات المعلومات من سمات GenAI.
- **استضافة ذاتية** — قم بتشغيل OTel Collector باستخدام معالج GenAI.

## Ship It

`outputs/skill-otel-genai.md` تمتد أسلاك OTel GenAI إلى وكيل موجود مع الإعدادات الافتراضية لالتقاط المحتوى والتخزين المرجعي الخارجي.

## Exercises

1. قم بتجهيز حلقة ReAct للدرس 01 باستخدام `invoke_agent` (داخلي) + امتدادات لكل أداة. إرسال إلى مثيل Jaeger.
2. قم بإضافة التقاط المحتوى في وضع "المراجع فقط": المطالبات إلى SQLite، وتحمل سمات الامتداد معرفات الصفوف فقط.
3. اقرأ مواصفات `gen_ai.data_source.id`. قم بتوصيله إلى بحث الدرس 09 Mem0.
4. قم بتعيين `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` وتأكد من عدم إعادة تسمية سماتك بواسطة المجمع.
5. أنشئ لوحة معلومات: "أخطاء الأداة التي ترتبط بالنماذج" من سمات GenAI وحدها.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| جيناي SIG | "مجموعة OpenTelemetry GenAI" | مجموعة عمل OTel تحدد المخطط |
| invoc_agent | "نطاق الوكيل" | اسم النطاق الذي يمثل تشغيل الوكيل |
| CLIENT مدى | "مكالمة عن بعد" | نطاق لاستدعاء خدمة وكيل بعيد |
| المدى الداخلي | "قيد المعالجة" | Span لتشغيل الوكيل قيد التشغيل |
| gen_ai.provider.name | "المزود" | أنثروبي / أوبناي / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG المصدر" | أي مجموعة/متجر تم استرجاعه |
| التقاط المحتوى | "التسجيل الفوري" | الاشتراك في التقاط الرسائل؛ تخزين خارجيا في همز |
| اختيار الاستقرار | "وضع المعاينة" | Env var لتثبيت الاتفاقيات التجريبية |

## Further Reading

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — the spec
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — امتدادات OTel مدمجة
- [كلود وكيل SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C تتبع انتشار السياق
