# لماذا متعدد الوكيل؟
> أحد العملاء يضرب الحائط. الخطوة الذكية ليست عميلاً أكبر، بل المزيد من العملاء.
**النوع:** تعلم
** اللغات: ** TypeScript
** المتطلبات الأساسية: ** المرحلة 14 (هندسة الوكيل)
**الوقت:** ~60 دقيقة
## أهداف التعلم
- تحديد سقف الوكيل الفردي (تجاوز السياق، الخبرة المختلطة، عنق الزجاجة المتسلسل) وشرح متى يكون التقسيم إلى وكلاء متعددين هو الخطوة الصحيحة
- قارن بين أنماط التنسيق (pipeline، والتوزيع المتوازي، والمشرف، والتسلسل الهرمي) وحدد النمط المناسب لهيكل مهمة معين
- تصميم نظام متعدد الوكلاء بحدود دور واضحة وحالة مشتركة وعقد اتصال
- تحليل المفاضلات بين تعقيد الوكيل المتعدد (زمن الوصول، والتكلفة، وصعوبة تصحيح الأخطاء) مقابل بساطة الوكيل الفردي
## المشكلة
لقد قمت ببناء وكيل واحد في المرحلة 14. وهو يعمل. يمكنه قراءة الملفات وتشغيل الأوامر واستدعاء واجهات برمجة التطبيقات والتفكير في النتائج. ثم تقوم بتوجيهه إلى قاعدة تعليمات برمجية حقيقية: 200 ملف، وثلاث لغات، واختبارات تعتمد على البنية التحتية، ومتطلبات البحث عن واجهات برمجة التطبيقات الخارجية قبل كتابة التعليمات البرمجية.
الوكيل يختنق. ليس لأن LLM غبي، ولكن لأن المهمة تتجاوز ما يمكن لحلقة وكيل واحدة التعامل معه. تمتلئ نافذة السياق بمحتويات الملف. ينسى الوكيل ما قرأه قبل 40 استدعاء للأداة. فهو يحاول أن يكون باحثًا، ومبرمجًا، ومراجعًا في الوقت نفسه، لكنه يفعل الثلاثة بشكل سيئ.
هذا هو سقف الوكيل الواحد. تضغط عليه في كل مرة تتطلب المهمة:
- **سياق أكثر مما يناسبه في نافذة واحدة** - قراءة 50 ملفًا تتجاوز 200 ألف رمز مميز
- **خبرات مختلفة في مراحل مختلفة** - يتطلب البحث تحفيزًا مختلفًا عن إنشاء التعليمات البرمجية
- **العمل الذي يمكن أن يتم بالتوازي** - لماذا تقرأ ثلاثة ملفات بالتتابع بينما يمكنك قراءتها في وقت واحد؟
##المفهوم
### سقف الوكيل الفردي
الوكيل الواحد عبارة عن حلقة واحدة ونافذة سياق واحدة وموجه نظام واحد. صورها:
```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

ثلاثة أشياء تنكسر:
1. **تشبع السياق** - تتراكم نتائج الأداة. بحلول الساعة 30، يكون الوكيل قد استهلك 150 ألف رمز مميز لمحتويات الملف ومخرجات الأوامر والاستدلال المسبق. تضيع تفاصيل مهمة من المنعطف الخامس.
2. **التباس الأدوار** - تؤدي مطالبة النظام التي تقول "أنت باحث ومبرمج ومراجع ومختبر" إلى إنتاج وكيل يقوم بنصف البحث ونصف التعليمات البرمجية ولا ينتهي من المراجعة أبدًا.
3. **عنق الزجاجة المتسلسل** - يقرأ الوكيل الملف A، ثم الملف B، ثم الملف C. ثلاث مكالمات تسلسلية LLM. ثلاث عمليات إعدام للأداة التسلسلية. لا التوازي.
### الحل متعدد الوكلاء
تقسيم العمل. امنح كل وكيل وظيفة واحدة، ونافذة سياق واحدة، وموجه نظام واحد تم ضبطه لهذه المهمة:
```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

كل وكيل لديه:
- موجه نظام مركّز ("أنت مراجع التعليمات البرمجية. ومهمتك الوحيدة هي العثور على الأخطاء.")
- نافذة السياق الخاصة بها (غير ملوثة بعمل الوكلاء الآخرين)
- عقد واضح للإدخال / الإخراج (يتلقى الملاحظات البحثية، رمز المخرجات)
### الأنظمة الحقيقية التي تفعل ذلك
**وكلاء Claude Code الفرعيون** - عندما يقوم Claude Code بإنشاء وكيل فرعي بـ `Task`، فإنه يقوم بإنشاء وكيل فرعي بمهمة محددة النطاق. يحافظ الوالد على سياقه نظيفًا. يقوم الطفل بعمل مركّز ويعيد ملخصًا.
**ديفين** - يدير وكيل مخطط، ووكيل مبرمج، ووكيل متصفح. يقسم المخطط العمل إلى خطوات. المبرمج يكتب التعليمات البرمجية. يبحث المتصفح في الوثائق. ولكل منها سياق منفصل.
**فرق الترميز متعددة الوكلاء (SWE-bench)** - تستخدم الأنظمة ذات الأداء الأفضل في SWE-bench باحثًا يقرأ قاعدة التعليمات البرمجية، ومخططًا يصمم الإصلاح، ومبرمجًا ينفذه. أنظمة الوكيل الفردي تسجل درجات أقل.
**ChatGPT Deep Research** - يولد العديد من وكلاء البحث بالتوازي، يستكشف كل منهم زاوية مختلفة، ثم يقوم بتجميع النتائج.
### الطيف
الوكيل المتعدد ليس ثنائيًا. وهو الطيف:
```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**وكيل واحد** - حلقة واحدة وموجه واحد. جيد للمهام البسيطة.
**الوكلاء الفرعيون** - يقوم أحد الوالدين بتوليد الأطفال للقيام بمهام فرعية مركزة. يحافظ الوالد على الخطة. تقرير الأطفال يعود. وهذا ما يفعله كلود كود.
**خط الأنابيب** - يعمل الوكلاء بالتسلسل. يصبح مخرجات الوكيل A مدخلات الوكيل B. جيد لسير العمل المرحلي: البحث -> الكود -> المراجعة -> الاختبار.
**الفريق** - يعمل الوكلاء بالتوازي مع ناقل الرسائل المشترك. ولكل منها دور. ينسق المنسق. جيد عندما تكون هناك حاجة إلى مهارات مختلفة في وقت واحد.
**السرب** - العديد من الوكلاء المتطابقين أو شبه المتطابقين ذوي الحالة المشتركة. لا يوجد منسق ثابت. يلتقط الوكلاء العمل من قائمة الانتظار. جيد للمهام الموازية عالية الإنتاجية.
### الأنماط الأربعة متعددة الوكلاء
#### النمط 1: خط الأنابيب
```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

يقوم كل وكيل بتحويل البيانات وتمريرها للأمام. بسيط للتفكير. الفشل في مرحلة واحدة يمنع الباقي.
#### النمط 2: مروحة للخارج/مروحة للداخل
```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

قم بتقسيم العمل عبر الوكلاء المتوازيين، ثم قم بدمج النتائج. جيد للمهام التي تتحلل إلى مهام فرعية مستقلة.
#### النمط 3: عامل منسق
```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

يقرر المنسق الذكي ما يجب فعله، ويفوض العمال، ويجمع النتائج. المنسق هو في حد ذاته وكيل لديه أدوات لتفريخ العمال.
#### النمط 4: سرب الأقران
```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

لا يوجد منسق مركزي. يتواصل الوكلاء من نظير إلى نظير. القرارات تنبثق من التفاعل. من الصعب تصحيح الأخطاء، ولكن يمكن تطبيقها على العديد من الوكلاء.
### متى NOT لاستخدام الوكيل المتعدد
يضيف الوكيل المتعدد التعقيد. كل رسالة بين الوكلاء هي نقطة فشل محتملة. ينتقل تصحيح الأخطاء من "قراءة محادثة واحدة" إلى "تتبع الرسائل عبر خمسة وكلاء".
**البقاء كوكيل واحد عندما:**
- يتم وضع المهمة في نافذة سياق واحدة (أقل من 100 ألف رمز مميز لبيانات العمل)
- لا تحتاج إلى مطالبات نظام مختلفة لمراحل مختلفة
- التنفيذ المتسلسل سريع بما فيه الكفاية
- المهمة بسيطة بما يكفي بحيث يؤدي تقسيمها إلى زيادة الحمل أكثر من القيمة
** تكلفة التعقيد: **
- كل حد للوكيل عبارة عن خطوة ضغط مع فقدان البيانات: يتم تلخيص السياق الكامل للوكيل A في رسالة للوكيل B
- منطق التنسيق (من يفعل ماذا ومتى وبأي ترتيب) هو مصدر الأخطاء الخاص به
- زيادة زمن الاستجابة: عدد N من الوكلاء يعني الحد الأدنى لعدد مكالمات LLM التسلسلية، وأكثر إذا كانوا بحاجة إلى التحدث ذهابًا وإيابًا
- تضاعف التكلفة: يقوم كل وكيل بحرق الرموز المميزة بشكل مستقل
القاعدة الأساسية: إذا كانت المهمة تتطلب أقل من 20 استدعاءًا للأداة وتناسب 100 ألف رمز مميز، فاحتفظ بها بوكيل واحد.
## بنائها
### الخطوة 1: الوكيل الفردي المثقل
هنا وكيل واحد يحاول أن يفعل كل شيء. يحتوي على موجه نظام ضخم ونافذة سياق واحدة تحتوي على الأبحاث والتعليمات البرمجية والمراجعات:
```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

مشاكل هذا النهج:
- نافذة السياق تنمو مع كل مرحلة. من خلال خطوة المراجعة، فإنه يحتوي على ملاحظات بحثية AND كود AND الاستدلال المسبق.
- موجه النظام عام. لا يمكن ضبطها لكل مرحلة.
- لا شيء يسير بالتوازي.
### الخطوة الثانية: الوكلاء المتخصصون
الآن تقسيمها. يحصل كل وكيل على وظيفة واحدة:
```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

كل متخصص لديه موجه مركزة. يحصل كل منها على نافذة سياق نظيفة تحتوي فقط على المدخلات التي يحتاجها.
### الخطوة 3: التنسيق عبر الرسائل
قم بتوصيل المتخصصين بتمرير رسالة صريحة:
```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

يتلقى كل وكيل فقط الرسائل الموجهة إليه. لا تلوث السياق. لا تدخل رموز قراءة التوثيق التي يبلغ عددها 50 ألفًا للباحث أبدًا في سياق المراجع.
### الخطوة الرابعة: المقارنة
```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

يستخدم الإصدار متعدد الوكلاء المزيد من الرموز المميزة (ثلاثة وكلاء، وثلاثة استدعاءات LLM منفصلة) ولكن يظل سياق كل وكيل نظيفًا. تتحسن جودة كل مرحلة لأن موجه النظام متخصص.
## استخدمه
يُنتج هذا الدرس مطالبة قابلة لإعادة الاستخدام لتحديد متى يجب الانتقال إلى عملاء متعددين. انظر `outputs/prompt-multi-agent-decision.md`.
## تمارين
1. أضف متخصصًا رابعًا: وكيل "مختبر" يتلقى التعليمات البرمجية من المبرمج ويراجع الملاحظات من المراجع، ثم يكتب الاختبارات
2. قم بتعديل سطر pipe حتى يتمكن المراجع من إرسال الملاحظات مرة أخرى إلى المبرمج للحصول على حلقة مراجعة (جولتان كحد أقصى)
3. قم بتحويل خط pipe المتسلسل إلى مخرج موسع: قم بتشغيل الباحث ووكيل "محلل المتطلبات" بالتوازي، ثم قم بدمج مخرجاتهما قبل التمرير إلى المبرمج
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| سرب | "عقل خلية وكلاء AI" | مجموعة من الوكلاء الأقران ذوي الحالة المشتركة وليس لديهم قائد ثابت. السلوك ينشأ من التفاعلات المحلية. |
| منسق | "العميل الرئيس" | وكيل تتضمن أدواته إنشاء وإدارة وكلاء آخرين. إنها تخطط وتفوض ولكنها قد لا تقوم بالعمل الفعلي. |
| منسق | "شرطي المرور" | مكون غير وكيل (غالبًا ما يكون مجرد رمز، وليس LLM) يقوم بتوجيه الرسائل بين الوكلاء بناءً على القواعد. |
| الإجماع | "الوكلاء متفقون" | بروتوكول حيث يجب أن يتوصل العديد من الوكلاء إلى اتفاق قبل المتابعة. يُستخدم عندما تحتاج المخرجات المتعارضة إلى حل. |
| السلوك الناشئ | "لقد اكتشف العملاء الأمر بأنفسهم" | الأنماط على مستوى النظام التي تنشأ من تفاعلات الوكيل ولكن لم تتم برمجتها بشكل صريح. يمكن أن تكون مفيدة أو ضارة. |
| مروحة للخارج/مروحة للداخل | "تصغير الخريطة للوكلاء" | تقسيم المهمة عبر وكلاء متوازيين (منتشر للخارج)، ثم دمج نتائجهم (مروحة للداخل). |
| تمرير الرسالة | "الوكلاء يتحدثون مع بعضهم البعض" | آلية الاتصال بين الوكلاء: البيانات المنظمة المرسلة من وكيل إلى آخر، لتحل محل نوافذ السياق المشتركة. |
## مزيد من القراءة
- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - مسح لأنماط الوكلاء المتعددين
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - إطار عمل المحادثة متعدد الوكلاء من Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - كيفية تفويض كلود كود للمهمة
- [CrewAI documentation](https://docs.crewai.com/) - إطار عمل متعدد الوكلاء قائم على الأدوار