# Natural Language Inference — Textual Entailment

> "t يستلزم h" يعني أن القراءة البشرية t ستستنتج أن h صحيح. NLI هي مهمة التنبؤ بالاستلزام / التناقض / الحياد. مملة على السطح، الحاملة في الإنتاج.

**النوع:** تعلم
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 05 (تحليل المشاعر)، المرحلة 5 · 13 (الإجابة على الأسئلة)
**الوقت:** ~60 دقيقة

## The Problem

لقد قمت ببناء ملخص. أنتجت ملخصا. كيف تعرف أن الملخص لا يحتوي على هلوسة؟

لقد قمت ببناء روبوت الدردشة. أجاب "نعم". كيف تعرف أن الإجابة مدعومة بالمقطع المسترجع؟

تحتاج إلى تصنيف 10000 مقالة إخبارية حسب الموضوع. ليس لديك أي تسميات التدريب. هل يمكنك إعادة استخدام النموذج؟

يتم تقليل جميع المشاكل الثلاثة إلى استنتاج اللغة الطبيعية. NLI يسأل: بالنظر إلى الفرضية `t` والفرضية `h`، هل `h` متضمنة في `t`، متناقضة، أو محايدة (غير مرتبطة)؟

- **فحص الهلوسة:** `t` = مستند المصدر، `h` = مطالبة ملخصة. لا يستلزم = الهلوسة.
- ** مؤرض QA:** `t` = مقطع مسترجع، `h` = إجابة تم إنشاؤها. لا استلزام = تلفيق.
- **تصنيف اللقطة الصفرية:** `t` = مستند، `h` = تسمية لفظية ("هذا يتعلق بالرياضة"). الاستحقاق = التسمية المتوقعة.

مهمة واحدة، ثلاثة استخدامات إنتاجية. هذا هو السبب في أن كل إطار تقييم RAG يشحن نموذج NLI تحت الغطاء.

## The Concept

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**المسميات الثلاثة.**

- **الاستلزم.** `t` → `h`. "القط على السجادة" يستلزم "هناك قطة".
- **التناقض.** `t` → ¬`h`. "القط على السجادة" يتناقض مع "ليس هناك قطة".
- **محايد.** لا يوجد استنتاج في كلتا الحالتين. "القطة على السجادة" محايدة لـ "القطة جائعة".

**ليس استدلالًا منطقيًا.** NLI هو استدلال لغوي *طبيعي* - وهو ما قد يستنتجه القارئ البشري العادي، وليس منطقًا صارمًا. "جون سار مع كلبه" يستلزم "جون لديه كلب" في NLI، لكن المنطق الصارم من الدرجة الأولى لن يعترف بذلك إلا إذا قمت بإضفاء طابع بديهي على الحيازة.

**Datasets.**

- **SNLI** (2015). 570 ألف زوج من التعليقات التوضيحية البشرية، والتعليقات التوضيحية للصور كمقرات عمل. المجال الضيق.
- **مالتي إنلي** (2017). 433 ألف زوجًا عبر 10 أنواع. مجموعة التدريب القياسية في عام 2026
- **ANLI** (2019). عدائية NLI. كتب البشر أمثلة مصممة خصيصًا لكسر النماذج الموجودة. أصعب.
- **DocNLI, ConTRoL** (2020–21). أماكن بطول الوثيقة. اختبارات القفزات المتعددة والاستدلال بعيد المدى.

**الهندسة المعمارية.** يقرأ برنامج تشفير المحولات (BERT، RoBERTa، DeBERTa) `[CLS] premise [SEP] hypothesis [SEP]`. يغذي التمثيل `[CLS]` softmax ثلاثي الاتجاهات. تدرب على MNLI، وقم بالتقييم بناءً على المعايير الثابتة، واحصل على دقة تزيد عن 90% في أزواج التوزيع.

**لقطة صفرية عبر NLI.** بالنظر إلى مستند وأسماء المرشحين، قم بتحويل كل تسمية إلى فرضية ("هذا النص يدور حول الرياضة"). حساب احتمالية الاستلزام لكل منها. اختر الحد الأقصى. هذه هي الآلية وراء خط Hugging Face `zero-shot-classification` pipe.

## Build It

### Step 1: run a pretrained NLI model

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

بالنسبة للإنتاج NLI، `facebook/bart-large-mnli` و `microsoft/deberta-v3-large-mnli` هي الإعدادات الافتراضية المفتوحة. يتصدر DeBERTa-v3 قوائم المتصدرين.

### Step 2: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

القالب هو "هذا المثال يدور حول {label}." بشكل افتراضي. التخصيص باستخدام `hypothesis_template`. لا توجد بيانات التدريب المطلوبة. لا صقل. يعمل خارج منطقة الجزاء.

### Step 3: faithfulness check for RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

هذا هو جوهر RAGAS الإخلاص. قم بتقسيم الإجابة الناتجة إلى مطالبات ذرية. تحقق من كل مطالبة مقابل السياق المسترد. قم بالإبلاغ عن الكسر الذي يستلزم ذلك.

### Step 4: hand-rolled NLI classifier (conceptual)

راجع `code/main.py` للحصول على لعبة stdlib فقط: تتم مقارنة الفرضية والفرضية من خلال التداخل المعجمي + اكتشاف النفي. ليست منافسة لنماذج المحولات - ولكنها تظهر شكل المهمة: نصان في، تسمية ثلاثية الاتجاه، الخسارة = إنتروبيا متقاطعة على `{entail, contradict, neutral}`.

## Pitfalls

- ** اختصارات الفرضية فقط. ** يمكن للنماذج التنبؤ بالتسمية من الفرضية وحدها بنسبة ~60% على SNLI لأن "لا"، "لا أحد"، "أبدًا" ترتبط بالتناقض. خط أساس قوي للكشف عن تسرب الملصقات.
- **إرشادي للتداخل المعجمي.** يمر إرشادي للتسلسل ("كل لاحقة") SNLI لكنه يفشل HANS/ANLI. استخدام معايير الخصومة.
- **تدهور طول المستند.** تسقط نماذج NLI ذات الجملة الواحدة 20+ F1 في المباني ذات طول المستند. استخدم النماذج المدربة بواسطة DocNLI للسياق الطويل.
- **حساسية قالب اللقطة الصفرية.** "هذا المثال يدور حول {label}" مقابل "{label}" مقابل "الموضوع هو {label}" يمكن أن يتأرجح الدقة بما يزيد عن 10 نقاط. ضبط القالب.
- **عدم تطابق المجال.** MNLI يتدرب على اللغة الإنجليزية العامة. يحتاج النص القانوني والطبي والعلمي إلى نماذج NLI خاصة بالمجال (على سبيل المثال، SciNLI، MedNLI).

## Use It

مكدس 2026:

| حالة الاستخدام | نموذج |
|---------|------|
| للأغراض العامة NLI | `microsoft/deberta-v3-large-mnli` |
| سريع / الحافة | `cross-encoder/nli-deberta-v3-base` |
| تصنيف الصفر شوت (خفيف) | `facebook/bart-large-mnli` |
| على مستوى الوثيقة NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| متعدد اللغات | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| كشف الهلوسة في RAG | NLI طبقة داخل RAGAS / DeepEval |

النمط الوصفي لعام 2026: NLI هو الشريط اللاصق لفهم النص. عندما تحتاج إلى "هل يدعم A B؟" أو "هل يتعارض أ مع ب؟" - اتصل بـ NLI قبل أن تصل لمكالمة LLM أخرى.

## Ship It

حفظ باسم `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Exercises

1. **سهل.** قم بتشغيل `facebook/bart-large-mnli` على 20 ثلاثية مصنوعة يدويًا (مقدمة، فرضية، تسمية) تغطي جميع الفئات الثلاثة. دقة القياس. أضف مصائد "الاستدلال اللاحق" الخصومة ("لم آكل الكعكة" مقابل "أكلت الكعكة") ومعرفة ما إذا كانت ستنكسر.
2. **متوسط.** قارن قالب اللقطة الصفرية `"This text is about {label}"` مع `"The topic is {label}"` و`"{label}"` في 100 AG عناوين الأخبار. تقرير تأرجح الدقة.
3. **صعب.** أنشئ مدقق الإخلاص RAG: تحليل المطالبات الذرية + NLI لكل مطالبة. قم بالتقييم على 50 إجابة تم إنشاؤها RAG ذات سياق ذهبي. قم بقياس المعدلات الإيجابية الكاذبة والسلبية الكاذبة مقابل الملصقات اليدوية.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| NLI | الاستدلال باللغة الطبيعية | 3-طريقة تصنيف العلاقة بين الفرضية والفرضية. |
| RTE | التعرف على المتضمن النصي | الاسم الأقدم لـ NLI؛ نفس المهمة. |
| استلزم | "ر يعني ح" | قد يستنتج القارئ النموذجي أن h صحيح بالنظر إلى t. |
| تناقض | "t يستبعد h" | قد يستنتج القارئ النموذجي أن h خطأ نظرًا لـ t. |
| محايد | "متردد" | لا يوجد استنتاج من t إلى h في كلتا الحالتين. |
| تصنيف صفر طلقة | NLI كمصنف | قم بصياغة التسميات لفظيًا على أنها فرضيات، واختر الحد الأقصى من التبعات. |
| الإخلاص | هل الجواب مدعوم؟ | NLI انتهى (السياق المسترجع، الإجابة المولدة). |

## Further Reading

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference]( — https.
- [Williams, Nangia, Bowman (2017). مجموعة تحديات واسعة النطاق لفهم الجملة من خلال الاستدلال](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — the ANLI benchmark.
- [Yin, Hay, Roth (2019). قياس تصنيف نص صفر طلقة](https://arxiv.org/abs/1909.00161) — NLI-كمصنف.
- [هو وآخرون. (2021). DeBERTa: فك التشفير المحسّن BERT مع الاهتمام المنفصل](https://arxiv.org/abs/2006.03654) — العمود الفقري لعام 2026 NLI.
