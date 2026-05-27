# Named Entity Recognition

> سحب الأسماء. يبدو الأمر سهلاً حتى تتعامل مع الحدود الغامضة والكيانات المتداخلة ومصطلحات المجال.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 5 · 02 (BoW + TF-IDF)، المرحلة 5 · 03 (تضمين الكلمات)
**الوقت:** ~75 دقيقة

## The Problem

"رفعت شركة Apple دعوى قضائية ضد Google بسبب صفقة بحث iPhone الخاصة بها في US." خمسة كيانات: أبل (ORG)، جوجل (ORG)، آيفون (PRODUCT)، صفقة البحث (ربما)، US (GPE). نظام NER جيد يستخرجها جميعًا بالأنواع الصحيحة. شخص سيء يفتقد iPhone، ويخلط بين Apple الفاكهة وApple الشركة، ويضع علامة "US" على أنها PERSON.

NER هو العمود الفقري تحت كل عملية استخلاص منظمة pipeline. تحليل السيرة الذاتية، ومسح سجل الامتثال، وإخفاء هوية السجلات الطبية، وفهم استعلام البحث، والأساس لاستجابات روبوت الدردشة، واستخراج العقود القانونية. أنت لا ترى ذلك أبدًا؛ أنت تعتمد عليه دائمًا.

يسير هذا الدرس في المسار الكلاسيكي (القائم على القواعد، HMM، CRF) إلى المسار الحديث (BiLSTM-CRF، ثم المحولات). تحل كل خطوة قيدًا محددًا للخطوة التي قبلها. النمط هو الدرس.

## The Concept

** BIO وضع العلامات ** (أو BILOU) يحول استخراج الكيان إلى مشكلة وضع العلامات على التسلسل. قم بتسمية كل رمز بـ `B-TYPE` (بداية الكيان)، أو `I-TYPE` (داخل الكيان)، أو `O` (خارج أي كيان).

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

سلسلة الكيانات متعددة الرموز: `New B-GPE`، `York I-GPE`، `City I-GPE`. النموذج الذي يفهم BIO يمكنه استخراج مسافات عشوائية.

تطور العمارة:

- **استنادًا إلى القواعد.** Regex + عمليات البحث في المعجم الجغرافي. دقة عالية في التعامل مع الكيانات المعروفة، وعدم وجود تغطية للكيانات الجديدة.
- **HMM.** نموذج ماركوف المخفي. احتمالية انبعاث الرمز المميز للعلامة المعطاة، واحتمالية الانتقال من علامة إلى أخرى. فك تشفير فيتربي. تم التدريب على البيانات المصنفة.
- **CRF.** حقل عشوائي مشروط. مثل HMM ولكنه تمييزي، بحيث يمكنك مزج ميزات عشوائية (شكل الكلمة، الكتابة بالأحرف الكبيرة، الكلمات المجاورة). لا يزال العمود الفقري للإنتاج الكلاسيكي في عام 2026 لعمليات النشر منخفضة الموارد.
- **BiLSTM-CRF.** ميزات عصبية بدلاً من الصنع اليدوي. LSTM يقرأ الجملة في كلا الاتجاهين، CRF طبقة في الأعلى تفرض تسلسلات متسقة للعلامات.
- **معتمد على المحولات.** ضبط دقيق BERT باستخدام رأس تصنيف الرمز المميز. أفضل دقة. معظم الحساب.

## Build It

### Step 1: BIO tagging helpers

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### Step 2: hand-crafted features

بالنسبة للكلاسيكية (غير العصبية) NER، الميزات هي اللعبة. مفيدة منها:

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` ترجع `xXxxxx`. `word_shape("USA-2024")` يعود `XXX-dddd`. أنماط الكتابة بالأحرف الكبيرة هي إشارة عالية للأسماء الصحيحة.

### Step 3: a simple rule-based + dictionary baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

تحتوي معاجم الإنتاج على ملايين الإدخالات المستخرجة من ويكيبيديا وDBpedia. التغطية جيدة. توضيح (`Apple` الشركة مقابل الفاكهة) أمر فظيع. ولهذا السبب فازت النماذج الإحصائية.

### Step 4: the CRF step (sketch, not full impl)

كامل CRF من الصفر في 50 سطرًا لا يكون مفيدًا بدون أسس نظرية الاحتمالية. استخدم `sklearn-crfsuite` بدلاً من ذلك:

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` و `c2` هما L1 و L2 التنظيم. `all_possible_transitions=True` يتيح للنموذج معرفة التسلسلات غير القانونية (على سبيل المثال، `I-ORG` بعد `O`) غير محتملة، وهذه هي الطريقة التي يفرض بها CRF الاتساق BIO دون كتابة القيد.

### Step 5: what a BiLSTM-CRF adds

الميزات تصبح مستفادة. المدخلات: تضمينات الرمز المميز (GloVe أو fastText). LSTM يقرأ من اليسار إلى اليمين ومن اليمين إلى اليسار. تمر الحالات المخفية المتسلسلة عبر طبقة إخراج CRF. لا يزال CRF يفرض اتساق تسلسل العلامات؛ يستبدل LSTM الميزات المصنوعة يدويًا بميزات تم تعلمها.

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

بالنسبة للطبقة CRF، استخدم `torchcrf.CRF` (pip تثبيت pytorch-crf). إن المكاسب التي تحققها من CRF المصنوعة يدويًا قابلة للقياس ولكنها أصغر مما تتوقع إلا إذا كان لديك عشرات الآلاف من الجمل المصنفة.

## Use It

يتم شحن SpaCy بدرجة إنتاجية NER خارج الصندوق.

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

لاحظ أن `iPhone` يحمل علامة `ORG` بدلاً من `PRODUCT` — يتميز نموذج SpaCy الصغير بتغطية ضعيفة لكيان المنتج. النموذج الكبير (`en_core_web_lg`) يعمل بشكل أفضل. نموذج المحول (`en_core_web_trf`) يعمل بشكل أفضل.

Hugging Face لـ BERT القائم على NER:

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` يدمج رموز B-X وI-X المتجاورة في نطاق. بدونها، ستحصل على تصنيفات على مستوى الرمز المميز وسيتعين عليك دمجها بنفسك.

### LLM-based NER (the 2026 option)

أصبحت الآن اللقطات الصفرية والقليلة LLM NER قادرة على المنافسة مع النماذج المضبوطة بدقة في العديد من المجالات، وهي أفضل بشكل كبير عندما تكون البيانات المصنفة نادرة.

- **مطالبة الصفر.** أعط LLM قائمة بأنواع الكيانات ومثال للمخطط. اطلب JSON الإخراج. يعمل خارج الصندوق؛ الدقة معتدلة في المجالات الجديدة.
- **المطالبة بأسلوب ZeroTuneBio.** قم بتحليل المهمة إلى استخراج المرشح ← تفسير المعنى ← الحكم ← إعادة التحقق. تعمل المطالبة متعددة المراحل (وليست طلقة واحدة) على رفع الدقة بشكل كبير في مجال الطب الحيوي NER. وينطبق نفس النمط على المجالات القانونية والمالية والعلمية.
- ** المطالبة الديناميكية باستخدام RAG.** استرجع الأمثلة الأكثر تشابهًا من مجموعة بذور مشروحة صغيرة لكل استدعاء استدلال؛ بناء موجه بضع طلقات على الطاير. في معايير عام 2026، يؤدي هذا إلى رفع GPT-4 الطب الحيوي NER F1 بنسبة 11-12% على المحفز الثابت.
- **تحليل لكل نوع كيان.** بالنسبة للمستندات الطويلة، يفقد استدعاء واحد يستخرج جميع أنواع الكيانات مرة واحدة تذكره مع زيادة الطول. تشغيل ممر استخراج واحد لكل نوع كيان. ارتفاع تكلفة الاستدلال، ودقة أعلى بكثير. هذا هو النمط القياسي للمذكرات السريرية والعقود القانونية.

توصية الإنتاج اعتبارًا من عام 2026: ابدأ بخط أساس LLM صفر قبل جمع بيانات التدريب. غالبًا ما يكون F1 جيدًا بدرجة كافية بحيث لا تحتاج أبدًا إلى الضبط الدقيق.

### Where classical NER still wins

حتى مع توفر LLMs، يفوز NER الكلاسيكي عندما:

- ميزانية الكمون أقل من 50 مللي ثانية.
- لديك الآلاف من الأمثلة المصنفة وتحتاج إلى 98%+ F1.
- يحتوي المجال على وجود مستقر حيث يتم نقل CRF أو BiLSTM المُدرب مسبقًا بشكل جيد.
- تتطلب القيود التنظيمية نموذجًا محليًا غير توليدي.

### Where it falls apart

- **تحويل المجال.** أداء CoNLL NER على العقود القانونية أسوأ من المعجم الجغرافي. ضبط المجال الخاص بك.
- **الكيانات المتداخلة.** "برج بنك أمريكا" هو ORG ومنشأة في نفس الوقت. لا يمكن أن يمثل المعيار BIO مسافات متداخلة. أنت بحاجة إلى NER متداخلة (نماذج متعددة التمريرات أو نماذج قائمة على الامتداد).
- **الكيانات الطويلة.** "المؤسسة الفيدرالية للتأمين على الودائع بالولايات المتحدة." أحيانًا تقسم النماذج على مستوى الرمز المميز هذا. استخدم `aggregation_strategy` أو ما بعد المعالجة.
- **أنواع متفرقة.** التصنيفات الطبية NER مثل DRUG_BRAND، ADVERSE_EVENT، DOSE. نماذج الأغراض العامة ليس لديها فكرة. Scispacy وBioBERT هما نقطتا البداية هناك.

## Ship It

حفظ باسم `outputs/skill-ner-picker.md`:

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## Exercises

1. **سهل.** قم بتنفيذ `bio_to_spans` (عكس `spans_to_bio`) وتحقق من اتساق الرحلة ذهابًا وإيابًا في 10 جمل.
2. **متوسط.** قم بتدريب sklearn-crfsuite CRF أعلاه على مجموعة بيانات CoNLL-2003 English NER. قم بالإبلاغ عن كل كيان F1 باستخدام `seqeval`. النتيجة النموذجية: ~84 F1.
3. **صعب.** الضبط الدقيق `distilbert-base-cased` على مجموعة بيانات NER خاصة بالمجال (طبية أو قانونية أو مالية). قارنه بالنموذج الصغير SpaCy. قم بتوثيق عمليات التحقق من تسرب البيانات واكتب ما فاجأك.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| NER | استخراج الأسماء | يمتد رمز التسمية إلى الأنواع (PERSON، ORG، GPE، DATE،...). |
| BIO | نظام وضع العلامات | `B-X` يبدأ، `I-X` يستمر، `O` بالخارج. |
| BILOU | أفضل BIO | يضيف `L-X` (الأخير)، `U-X` (وحدة) لحدود أكثر وضوحًا. |
| CRF | المصنف المنظم | نماذج التحولات بين التسميات، وليس فقط الانبعاثات. يفرض تسلسلات صالحة. |
| متداخلة NER | الكيانات المتداخلة | الامتداد الواحد هو كيان مختلف عن الامتداد الفرعي له. BIO لا يستطيع التعبير عن هذا. |
| على مستوى الكيان F1 | السليم NER متري | يجب أن يتطابق النطاق المتوقع مع النطاق الحقيقي تمامًا. مستوى الرمز المميز F1 يبالغ في الدقة. |

## Further Reading

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — the BiLSTM-CRF paper. Canonical.
- [Devlin et al. (2018). BERT: التدريب المسبق للمحولات العميقة ثنائية الاتجاه](https://arxiv.org/abs/1810.04805) — يقدم نمط تصنيف الرمز المميز الذي أصبح قياسيًا.
- [الميزات اللغوية SpaCy - الكيانات المسماة](https://spacy.io/usage/linguistic-features#named-entities) - مرجع عملي لكل سمة في `Doc.ents` و`Span`.
- [seqeval](https://githubhub.com/chakki-works/seqeval) — مكتبة المقاييس الصحيحة. استخدمها دائمًا.
