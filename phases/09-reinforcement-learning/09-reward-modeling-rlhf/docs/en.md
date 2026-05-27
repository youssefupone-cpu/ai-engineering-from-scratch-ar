# نمذجة المكافآت و RLHF
> لا يستطيع البشر كتابة دالة مكافأة لـ "الاستجابة المساعدة الجيدة"، ولكن يمكنهم مقارنة استجابتين واختيار الأفضل. قم بملاءمة نموذج المكافأة مع تلك المقارنات، ثم RL نموذج اللغة مقابله. كريستيانو 2017. InstructGPT 2022. الوصفة التي حولت GPT-3 إلى ChatGPT. في عام 2026، سيتم استبداله في الغالب بـ DPO - لكن النموذج العقلي يظل قائمًا.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 05 (المشاعر)، المرحلة 9 · 08 (PPO)
**الوقت:** ~45 دقيقة
## المشكلة
لقد قمت بتدريب نموذج لغة على هدف التنبؤ بالرمز المميز التالي. ويكتب قواعد اللغة الإنجليزية. كما أنه يكذب ويتعثر ويرفض الرفض. لا يمكنك إصلاح ذلك بمزيد من التدريب المسبق - نص الويب هو المشكلة وليس العلاج.
تريد *مكافأة عددية* تقول "الإجابة أ أفضل من الإجابة ب للتعليمات X." إن كتابة وظيفة المكافأة يدويًا أمر مستحيل. "المساعدة" ليست تعبيرًا مغلقًا على الرموز المميزة. لكن يمكن للبشر مقارنة مخرجين وتحديد التفضيل. وهذا أمر رخيص للجمع على نطاق واسع.
RLHF (Christiano et al. 2017; Ouyang et al. 2022) يحول التفضيلات إلى نموذج مكافأة، ثم يقوم بتحسين LM عبر PPO مقابل تلك المكافأة. في ثلاث خطوات: SFT → RM → PPO. إنها الوصفة التي شحنت ChatGPT وClaude وGemini وكل الآخرين المتوافقين مع LLM في 2023-2025.
في عام 2026، يتم استبدال خطوة PPO في الغالب بـ DPO (المرحلة 10 · 08) لأنها أرخص وجيدة تقريبًا لضبط المحاذاة. لكن قطعة *نموذج المكافأة* لا تزال تكمن وراء كل عينة من أفضل ما في N، وكل سطر RL-from-verifiable-rewards pipe، وكل نموذج تفكير يستخدم نموذج مكافأة العملية. افهم RLHF وستفهم مجموعة المحاذاة بأكملها.
##المفهوم
![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)
**المرحلة 1: الضبط الدقيق تحت الإشراف (SFT).** ابدأ من نموذج أساسي تم تدريبه مسبقًا. صقل العروض التوضيحية المكتوبة بواسطة الإنسان للسلوك المستهدف (الاستجابات التي تتبع التعليمات، والردود المفيدة، وما إلى ذلك). النتيجة: نموذج `π_SFT` *متحيز نحو السلوك الجيد* ولكن لا يزال لديه مساحة عمل غير محدودة.
**المرحلة الثانية: التدريب على نموذج المكافأة.**
- اجمع أزواجًا من الاستجابات `(y_+, y_-)` للمطالبات `x`، التي صنفها البشر على أنها "y_+ مفضلة على y_-."
- قم بتدريب نموذج المكافأة `R_φ(x, y)` لتعيين درجات أعلى لـ `y_+`.
- الخسارة: **الثنائية اللوجستية بين برادلي وتيري**:
__الكود_0__
σ هو السيني. يشير الاختلاف في المكافأة إلى احتمالات التفضيل. BT هو المعيار منذ عام 1952 (Bradley-Terry) وهو الاختيار السائد في RLHF الحديث.
- تتم تهيئة `R_φ` عادةً من النموذج SFT برأس عددي في الأعلى. نفس العمود الفقري للمحولات. طبقة خطية واحدة تنتج المكافأة.
**المرحلة 3: PPO مقابل RM مع عقوبة KL.**
- تهيئة السياسة القابلة للتدريب `π_θ` من `π_SFT`. احتفظ بـ *مرجع* مجمد `π_ref = π_SFT`.
- المكافأة في نهاية الرد `y`:
__الكود_0__
تمنع عقوبة KL `π_θ` من الانجراف بشكل تعسفي من `π_SFT` - إنها *منظم*، وليست منطقة ثقة صعبة. `β` عادةً `0.01`-`0.05`.
- قم بتشغيل PPO (الدرس 08) بهذه المكافأة. يتم حساب المزايا على المسار على مستوى الرمز المميز، لكن RM يسجل الاستجابة الكاملة فقط.
**لماذا KL؟** بدونه، سيجد PPO بكل سرور إستراتيجيات اختراق المكافآت - تم تدريب RM فقط على عمليات الإكمال أثناء التوزيع. قد تحصل الاستجابة خارج التوزيع على درجات أعلى من أي استجابة مكتوبة بواسطة الإنسان. يحتفظ KL بـ `π_θ` بالقرب من المشعب حيث تم تدريب RM. إنه المقبض الوحيد الأكثر أهمية في RLHF.
**الحالة 2026:**
- **DPO** (رافايلوف 2023): ينهار الجبر ذو الشكل المغلق المرحلة 2+3 إلى خسارة واحدة خاضعة للإشراف على بيانات التفضيل. لا RM، لا PPO. نفس الجودة في معايير المحاذاة لجزء صغير من الحساب. تمت تغطيته في المرحلة 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO مع خط أساس متعلق بالمجموعة بدلاً من الناقد، مكافأة من *المدقق* (تشغيل التعليمات البرمجية / مطابقة الإجابات الرياضية) بدلاً من RM المدرب بشريًا. المهيمنة على نماذج الاستدلال. تمت تغطيته في المرحلة 9 · 12.
- **نماذج مكافآت العمليات (PRMs):** درجات الحلول الجزئية (كل خطوة من خطوات الاستدلال)، المستخدمة في كل من متغيرات RLHF وGRPO للاستدلال.
- **الدستوري AI / RLAIF:** استخدم LLM لإنشاء التفضيلات بدلاً من البشر. يقيس ميزانية التفضيل.
## بنائها
يستخدم هذا الدرس "مطالبات" و"استجابات" اصطناعية صغيرة يتم تمثيلها على شكل سلاسل. RM عبارة عن مسجل خطي فوق تمثيل حقيبة الرموز المميزة. لا يوجد LLM حقيقي — شكل *الخط pipe مهم، وليس المقياس. انظر `code/main.py`.
### الخطوة 1: بيانات التفضيلات التركيبية
```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

في RLHF الحقيقي، يتم استبدال هذا بواضعي العلامات البشرية. الشكل — `(prompt, preferred_response, rejected_response)` — متطابق.
### الخطوة الثانية: نموذج مكافأة برادلي-تيري
النتيجة الخطية: `R(x, y) = w · bag(y)`. تدريب لتقليل فقدان السجل الزوجي BT:
```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

بعد بضع مئات من التحديثات، يقوم `w` بتعيين أوزان إيجابية لرموز الكلمات الجيدة وسالبة للكلمات السيئة.
### الخطوة 3: سياسة الإعجاب بـ PPO أعلى RM
تنتج سياسة الألعاب الخاصة بنا رمزًا واحدًا من المفردات. نحن نسجل الرمز المميز تحت RM، ونحسب `log π_θ(token | prompt)`، ونضيف عقوبة KL إلى المرجع، ونطبق البديل المقطوع PPO.
```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### الخطوة 4: مراقبة KL
تتبع يعني `KL(π_θ || π_ref)` كل تحديث. إذا تجاوزت `~5-10`، فقد انحرفت السياسة بعيدًا عن `π_SFT` — `β` الأدنى آخذ في الارتفاع أو بدأ اختراق المكافآت. هذا هو التشخيص الأفضل في الواقع RLHF.
### الخطوة 5: وصفة الإنتاج باستخدام TRL
بمجرد أن تفهم اللعبة pipeline، إليك نفس الحلقة التي يكتبها مستخدم المكتبة الحقيقي. Hugging Face's [TRL](https://huggingface.co/docs/trl) هو التنفيذ المرجعي — `RewardTrainer` للمرحلة 2 و `PPOTrainer` (مع KL-to-reference مضمن) للمرحلة 3.
```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

ثلاثة أشياء تفعلها المكتبة من أجلك. `adap_kl_ctrl=True` ينفذ جدول β التكيفي: إذا تمت ملاحظة KL يتجاوز `target_kl`، فإن β يتضاعف؛ إذا كان أقل من النصف، β نصفين. تم تجميد النموذج المرجعي حسب التقليد — يجب ألا تقوم بمشاركة المعلمات عن طريق الخطأ مع `policy`. ويعيش رأس القيمة على نفس العمود الفقري للسياسة (`AutoModelForCausalLMWithValueHead` يرفق رأسًا عدديًا MLP)، ولهذا السبب يقوم TRL بإبلاغ `policy/kl` و`value/loss` بشكل منفصل.
## مطبات
- **الإفراط في التحسين / اختراق المكافآت.** RM غير كامل؛ يجد `π_θ` عمليات إكمال تنافسية ذات نقاط عالية ولكنها سيئة. الأعراض: تصعد المكافأة إلى أجل غير مسمى بينما يسجل تقييم الإنسان ثباتًا أو انخفاضًا. الإصلاح: التوقف مبكرًا، ورفع `β`، وتوسيع RM بيانات التدريب.
- **اختراق الطول.** غالبًا ما يكافئ مديرو العلاقات الذين تم تدريبهم على الاستجابات المفيدة ضمنًا الطول. تتعلم السياسة كيفية وضع الاستجابات. العلاج: مكافأة ذات طول طبيعي، أو RLAIF مع RM مع مراعاة الطول.
- **صغير جدًا RM.** يجب أن يكون RM بحجم السياسة على الأقل. لا يمكن لـ RM الصغير أن يسجل نتائج السياسة بدقة.
- ضبط **KL.** منخفض جدًا β → الانجراف ومكافأة القرصنة. سياسة β → مرتفعة للغاية بالكاد تتغير. الحيلة القياسية هي *تكيفية* β تستهدف KL ثابتًا في كل خطوة.
- **ضوضاء البيانات المفضلة.** ~30% من التسميات البشرية مزعجة أو غامضة. قم بالمعايرة من خلال تدريب RM على البيانات التي تمت تصفيتها بموجب الاتفاقية أو استخدم درجة الحرارة في BT.
- **مشاكل خارج السياسة.** بيانات PPO خارجة عن السياسة قليلاً بعد الفترة الأولى. مراقبة جزء المقطع كما في الدرس 08.
## استخدمه
RLHF في عام 2026 متعدد الطبقات:
| طبقة | الهدف | الطريقة |
|-------|--------|--------|
| التعليمات التالية، المساعدة، عدم الضرر | محاذاة | DPO (المرحلة 10 · 08) مفضل على RLHF-PPO. |
| صحة الاستدلال (الرياضيات، الكود) | القدرة | GRPO مع مكافأة التحقق (المرحلة 9 · 12). |
| مهام متعددة الخطوات طويلة الأفق | وكيل | PPO / GRPO مع نماذج مكافأة العملية عبر الخطوات. |
| سلوك الأمان/الرفض | السلامة | RLHF-PPO مع سلامة منفصلة RM، أو AI دستورية. |
| أفضل ما في N عند الاستدلال | محاذاة سريعة | استخدم RM في وقت فك التشفير؛ لا حاجة للتدريب على السياسات. |
| مكافأة التقطير | حساب الاستدلال | قم بتدريب "رأس مكافأة" صغير فوق LM مجمد. |
RLHF كان *الأسلوب* في الفترة 2022-2024. في عام 2026، تكون خطوط محاذاة الإنتاج pip هي DPO-الأولى، PPO-فقط للخطوات RM المكثفة أو الحرجة للسلامة.
## اشحنها
حفظ باسم `outputs/skill-rlhf-architect.md`:
```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## تمارين
1. **سهل.** قم بتدريب نموذج مكافأة برادلي-تيري في `code/main.py` على 500 زوج مفضل اصطناعي. قم بقياس الدقة الزوجية على 100 زوج معلق. يجب أن تتجاوز 90%.
2. **متوسط.** قم بتشغيل حلقة اللعبة PPO-RLHF باستخدام `β ∈ {0.0, 0.1, 1.0}`. لكل منها، قم برسم نقاط RM مقابل KL للإشارة إلى التحديثات. الذي يدير مكافأة الإختراق؟
3. **صعب.** تنفيذ DPO (خسارة احتمالية التفضيل في النموذج المغلق) على نفس بيانات التفضيل ومقارنتها بخط RLHF-PPO pipe في الحساب المستخدم والنتيجة النهائية RM التي تم تحقيقها.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| __المصطلح_2__ | "المحاذاة RL" | ثلاث مراحل SFT + RM + PPO pipeline (كريستيانو 2017، أويانغ 2022). |
| نموذج المكافأة (RM) | "شبكة التهديف" | تتناسب الدالة العددية المستفادة مع التفضيلات الزوجية عبر برادلي تيري. |
| برادلي تيري | "الخسارة اللوجستية الزوجية" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; الهدف RM القياسي. |
| KL عقوبة | "كن بالقرب من المرجع" | `β · KL(π_θ || π_ref)` في المكافأة؛ منظم مكافحة القرصنة والمكافأة. |
| مكافأة القرصنة | "قانون جودهارت" | تستغل السياسة عيوب RM؛ الأعراض: مكافأة تصل، تقييم الإنسان شقة. |
| RLAIF | "AI-التفضيلات المسماة" | RLHF حيث تأتي التسميات من LM آخر بدلاً من البشر. |
| PRM | "نموذج مكافأة العملية" | يسجل خطوات التفكير الجزئي؛ المستخدمة في الاستدلال pipelines. |
| دستوري __المصطلح_16__ | "المنهج الأنثروبي" | AI-التفضيلات التي تم إنشاؤها مسترشدة بقواعد صريحة. |
## مزيد من القراءة
- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — الورقة التي بدأت RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — الوصفة وراء ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — سابقًا RLHF للتلخيص.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO؛ الافتراضي بعد RLHF في عام 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF وحلقة النقد الذاتي.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — ورقة HH.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — الإنتاج `RewardTrainer` و `PPOTrainer`. اقرأ مصدر المدرب للحصول على KL التكيفي وتفاصيل القيمة الرئيسية.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) بواسطة لامبرت، كاستريكاتو، فون ويرا، هافريلا - العرض الأساسي لخط pipe المكون من ثلاث مراحل مع الرسوم البيانية.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — المكتبة؛ يحتوي `examples/` على نصوص RLHF شاملة لكل من Llama وMistral وQwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — عرض فرضية المكافأة؛ شرط أساسي للتفكير في قرصنة المكافأة.