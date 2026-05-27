# OpenTelemetry GenAI — Tracing Tool Calls End-to-End

> يقوم الوكيل باستدعاء خمس أدوات وثلاثة خوادم MCP ووكيلين فرعيين. أنت بحاجة إلى أثر واحد عبر كل ذلك. تعد الاصطلاحات الدلالية OpenTelemetry GenAI (السمات الثابتة في الإصدار 1.37 والإصدارات الأحدث) معيار 2026، وهي مدعومة أصلاً بواسطة Datadog وLangfuse وArize Phoenix وOpenLLMetry وAgentOps. يسمي هذا الدرس السمات المطلوبة، ويسير في التسلسل الهرمي للامتداد (الوكيل → LLM → الأداة)، ويشحن باعث نطاق stdlib الذي يمكنك توصيله بأي مصدر لـ OTel.

**النوع:** بناء
** اللغات: ** بايثون (stdlib، OTelspan emitter)
**المتطلبات:** المرحلة 13 · 07 (MCP خادم)، المرحلة 13 · 08 (MCP عميل)
**الوقت:** ~75 دقيقة

## Learning Objectives

- قم بتسمية سمات OTel GenAI المطلوبة لامتداد LLM ونطاق تنفيذ الأداة.
- قم ببناء تسلسل هرمي للتتبع يغطي حلقة الوكيل، واستدعاء LLM، واستدعاء الأداة، وMCP إرسال العميل.
- حدد المحتوى الذي تريد التقاطه (الاشتراك) مقابل التنقيح (الإعدادات الافتراضية).
- إرسال الامتدادات إلى المجمع المحلي (Jaeger، Langfuse) دون إعادة كتابة كود الأداة.

## The Problem

تصحيح أخطاء من فبراير 2026: أبلغ المستخدم أن "وكيلي يستغرق أحيانًا 30 ثانية للرد؛ وفي أحيان أخرى 3 ثوانٍ." لا آثار. تُظهر السجلات المكالمة LLM، ولكن ليس إرسال الأداة، وليس ذهابًا وإيابًا للخادم MCP، وليس الوكيل الفرعي. هل تخمن. في النهاية تجد: خادم MCP واحد يتعطل أحيانًا عند بداية التشغيل البارد.

بدون التتبع الشامل، لا يمكنك العثور على هذا. يقوم OTel GenAI بإصلاحه.

استقرت الاتفاقيات في 2025-2026 ضمن مجموعة الاتفاقيات الدلالية OpenTelemetry. وهي تحدد أسماء السمات الثابتة، لذا تقوم كل من Datadog وLangfuse وPhoenix وOpenLLMetry وAgentOps بتحليل نفس النطاقات. الصك مرة واحدة؛ السفينة إلى أي الخلفية.

## The Concept

### Span hierarchy

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

كل شيء يعشش تحت معرف تتبع واحد. معرفات الامتداد تربط العلاقات بين الوالدين والطفل.

### Required attributes

وفقًا لـ semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`، `"text_completion"`، `"embeddings"`، `"execute_tool"`، `"invoke_agent"`.
- `gen_ai.provider.name` — toolidentifier، ، ، .
- `gen_ai.request.model` — سلسلة النموذج المطلوبة (على سبيل المثال `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — النموذج يخدم بالفعل.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — معرف استجابة الموفر للارتباط.

بالنسبة لامتدادات الأداة:

- `gen_ai.tool.name` — معرف الأداة.
- `gen_ai.tool.call.id` — معرف المكالمة المحدد.
- `gen_ai.tool.description` — وصف الأداة (اختياري).

بالنسبة لامتدادات الوكيل:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Span kinds

- `SpanKind.CLIENT` للمكالمات التي تتجاوز حدود العملية (مزود LLM، خادم MCP).
- `SpanKind.INTERNAL` لخطوات الحلقة الخاصة بالوكيل وتنفيذ الأداة.

### Opt-in content capture

بشكل افتراضي، تحمل الامتدادات مقاييس وتوقيتًا - وليس المطالبات أو الإكمالات. يتم إيقاف الحمولات الكبيرة وPII بشكل افتراضي. قم بتعيين `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` ومتغيرات env محددة لالتقاط المحتوى لتضمين المحتوى. راجع بعناية قبل التمكين في همز.

### Events on spans

يمكن إضافة الأحداث على مستوى الرمز المميز كأحداث ممتدة:

- `gen_ai.content.prompt` — رسائل الإدخال.
- `gen_ai.content.completion` — رسائل الإخراج.
- `gen_ai.content.tool_call` — استدعاء الأداة كما هو مسجل.

ترتيب زمني للأحداث خلال فترة زمنية لإعادة التشغيل بشكل تفصيلي.

### Exporters

يمتد نطاق OTel إلى:

- **جايجر / تيمبو.** OSS، محليًا.
- **Langfuse.** LLM-خاص بقابلية الملاحظة؛ يتصور استخدام الرمز المميز.
- **Arize Phoenix.** التقييمات + التتبع مجتمعة.
- **Datadog.** تجاري؛ يوزع أصلاً `gen_ai.*` السمات.
- **قرص العسل.** موجه نحو العمود؛ سهل الاستعلام.

الكل يتحدث OTLP، تنسيق السلك. الكود الخاص بك لا يهتم.

### Propagation across MCP

عندما يتصل عميل MCP بالخادم، أدخل رأس التتبع W3C في الطلب. يدعم البث HTTP الرؤوس القياسية. Stdio لا يحمل رؤوس HTTP أصلاً؛ تناقش خريطة الطريق الخاصة بالمواصفات لعام 2026 إضافة حقل `_meta.traceparent` للمكالمات JSON-RPC.

حتى يتم الشحن: قم بتضمين التتبع في `_meta` لكل طلب يدويًا. يسجل الخادم معرف التتبع.

### Metrics

إلى جانب الامتدادات، يحدد GenAI semconv المقاييس:

- `gen_ai.client.token.usage` — الرسم البياني.
- `gen_ai.client.operation.duration` — الرسم البياني.
- `gen_ai.tool.execution.duration` — رسم بياني.

استخدمها للوحات المعلومات التي لا تحتاج إلى تفاصيل لكل مكالمة.

### AgentOps layer

AgentOps (تأسست عام 2024) متخصصة في إمكانية ملاحظة GenAI. فهو يغلف الأطر الشائعة (LangGraph، Pydantic AI، CrewAI) لإصدار امتدادات OTel تلقائيًا. مفيد إذا كانت مجموعتك تستخدم إطار عمل مدعومًا؛ استخدام الأجهزة اليدوية خلاف ذلك.

## Use It

`code/main.py` يصدر امتدادات على شكل OTel إلى stdout (بتنسيق يشبه OTLP-JSON) للوكيل الذي يستدعي LLM، ويرسل أداتين، وmakes واحدة MCP ذهابًا وإيابًا. لا يوجد مصدر حقيقي - يركز الدرس على شكل الامتداد ومجموعة السمات. الصق الإخراج في عارض متوافق مع OTLP أو قم بقراءته فقط.

ما الذي يجب النظر إليه:

- تتم مشاركة معرف التتبع عبر جميع الامتدادات.
- يتم تشفير الروابط بين الوالدين والطفل عبر `parentSpanId`.
- يتم ملء السمات `gen_ai.*` المطلوبة.
- يتم إيقاف التقاط المحتوى بشكل افتراضي؛ يقوم أحد السيناريوهات بتشغيله عبر env var.

## Ship It

ينتج عن هذا الدرس `outputs/skill-otel-genai-instrumentation.md`. بالنظر إلى قاعدة بيانات الوكيل، تنتج المهارة خطة أدوات: مكان إضافة الامتدادات، والسمات التي يجب نشرها، والمصدرين الذين يجب استهدافهم.

## Exercises

1. قم بتشغيل `code/main.py`. قم بعد المسافات وحدد ما هو CLIENT مقابل INTERNAL.

2. قم بتشغيل التقاط المحتوى (env var) وتأكد من ظهور الأحداث `gen_ai.content.prompt` و`gen_ai.content.completion`. لاحظ الآثار المترتبة على PII.

3. أضف مقياس تنفيذ الأداة `gen_ai.tool.execution.duration` وأرسله كعينة رسم بياني لكل مكالمة.

4. قم بنشر التتبع من امتداد الوكيل الأصلي إلى الحقل `_meta.traceparent` الخاص بالطلب MCP. تأكد من أن الخادم MCP سيشاهد نفس معرف التتبع.

5. اقرأ مواصفات OTel GenAI semconv. حدد سمة واحدة مدرجة في semconv التي ينبعث منها كود هذا الدرس NOT. أضفه.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| أوتيل | "القياس عن بعد المفتوح" | المعيار المفتوح للآثار والمقاييس والسجلات |
| GenAI semconv | "الاصطلاحات الدلالية GenAI" | أسماء السمات الثابتة لـ LLM / الأداة / الوكيل تمتد |
| `gen_ai.*` | "مساحة اسم السمة" | تشترك جميع سمات GenAI في هذه البادئة |
| سبان | "عملية موقوتة" | وحدة عمل لها بداية ونهاية وسمات |
| تتبع | "النسب عبر الامتداد" | شجرة الامتدادات تتقاسم معرف التتبع |
| سبانكايند | "CLIENT / SERVER / داخلي" | تلميحات حول اتجاه الامتداد |
| OTLP | "بروتوكول خط القياس عن بعد المفتوح" | تنسيق السلك للمصدرين |
| محتوى الاشتراك | "التقاط المطالبة/الإكمال" | معطل بشكل افتراضي؛ env var لتمكين |
| التتبع | "W3C رأس" | نشر سياق التتبع عبر الخدمات |
| مصدر | "الشاحن الخاص بالواجهة الخلفية" | المكون الذي يرسل الامتدادات إلى Jaeger / Datadog / إلخ. |

## Further Reading

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — الاصطلاحات الأساسية لامتدادات GenAI ومقاييسها وأحداثها
- [OpenTelemetry — يمتد GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM وقائمة سمات مدى تنفيذ الأداة
- [OpenTelemetry — امتدادات وكيل GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — مستوى الوكيل `invoke_agent` امتداد
- [القياس عن بعد المفتوح/الاصطلاحات الدلالية — امتدادات GenAI](https://githubhub.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub-مصدر مستضاف للحقيقة
- [Datadog — LLM الاصطلاح الدلالي لـ OTel](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — إرشادات تكامل الإنتاج
