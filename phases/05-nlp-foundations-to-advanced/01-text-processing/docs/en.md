# Text Processing — Tokenization, Stemming, Lemmatization

> اللغة مستمرة. النماذج منفصلة. المعالجة المسبقة هي الجسر.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 2 · 14 (نايف بايز)
**الوقت:** ~45 دقيقة

## The Problem

لا يمكن للنموذج قراءة "كانت القطط تجري". يقرأ الأعداد الصحيحة.

يفتح كل نظام NLP بنفس الأسئلة الثلاثة. أين تبدأ الكلمة. ما هو جذر الكلمة. كيف نتعامل مع "تشغيل"، "تشغيل"، "تشغيل" على أنها نفس الشيء عندما يكون ذلك مفيدًا، وكأشياء مختلفة عندما لا يكون ذلك مفيدًا.

أخطأ في الترميز وسيتعلم النموذج من البيانات المهملة. إذا كان برنامج الرمز المميز الخاص بك يعامل `don't` كرمز مميز واحد ولكن `do n't` كرمزين، فسيتم تقسيم توزيع التدريب. إذا انهار جذعك `organization` و`organ` على نفس الجذع، فإن نموذج الموضوع يموت. إذا كان lemmatizer الخاص بك يحتاج إلى سياق جزء من الكلام ولكنك لا تمرره، فسيتم التعامل مع الأفعال كأسماء.

يبني هذا الدرس أساسيات المعالجة المسبقة الثلاثة من الصفر، ثم يوضح كيف يقوم NLTK وspaCy بنفس العمل حتى تتمكن من رؤية المفاضلات.

## The Concept

ثلاث عمليات. لكل منها وظيفة ووضع الفشل.

**الترميز** يقسم السلسلة إلى رموز مميزة. "الرمز المميز" غامض بشكل متعمد لأن الدقة الصحيحة تعتمد على المهمة. مستوى الكلمة للكلاسيكية NLP. الكلمة الفرعية للمحولات. حرف للغات بدون مسافات بيضاء.

**جذع** لواحق القطع مع القواعد. سريع، عدواني، غبي. `running -> run`. `organization -> organ`. هذا الثاني هو وضع الفشل.

** Lemmatization ** يقلل الكلمة إلى شكل القاموس الخاص بها باستخدام المعرفة النحوية. أبطأ ودقيق ويحتاج إلى جدول بحث أو محلل صرفي. `ran -> run` (يحتاج إلى معرفة أن "run" هو زمن الماضي من "run"). `better -> good` (يحتاج إلى معرفة أشكال المقارنة).

القاعدة الأساسية. توقف عندما تكون السرعة مهمة ويمكنك تحمل الضوضاء (فهرسة البحث، التصنيف التقريبي). تكلم عندما يكون المعنى مهمًا (الإجابة على الأسئلة، البحث الدلالي، أي شيء سيقرأه المستخدم).

## Build It

### Step 1: a regex word tokenizer

أبسط أداة رمزية مفيدة تنقسم إلى أحرف غير أبجدية رقمية مع الاحتفاظ بعلامات الترقيم كرموز خاصة بها. ليست مثالية، وليست نهائية، ولكنها تسير في سطر واحد.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

ثلاثة أنماط حسب الأسبقية. الكلمات ذات الفاصلة العليا الداخلية الاختيارية (`don't`، `it's`). أرقام نقية. أي حرف مفرد بدون مسافات بيضاء وغير أبجدية رقمية كرمز مميز مستقل (علامات الترقيم).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

أوضاع الفشل للملاحظة. `3pm` ينقسم إلى `['3', 'pm']` لأننا قمنا بالتناوب بين تشغيل الحروف وتشغيل digit. جيد بما فيه الكفاية لمعظم المهام. تنكسر عناوين URL ورسائل البريد الإلكتروني وعلامات التصنيف. بالنسبة للإنتاج، قم بإضافة أنماط قبل الأنماط العامة.

### Step 2: a Porter stemmer (step 1a only)

تحتوي خوارزمية بورتر الكاملة على خمس مراحل من القواعد. تغطي الخطوة 1 أ وحدها اللواحق الإنجليزية الأكثر شيوعًا وتعلم النمط.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

اقرأ القواعد من أعلى إلى أسفل. القاعدة `ies -> i` هي لماذا `ponies -> poni`، وليس `pony`. يحتوي Real Porter على الخطوة 1 ب التي من شأنها إصلاح المشكلة. تتنافس القواعد. القواعد السابقة تفوز. الترتيب يهم أكثر من أي قاعدة واحدة.

### Step 3: a lookup-based lemmatizer

Lemmatization الاحتياجات المناسبة التشكل. تستخدم النسخة التعليمية القابلة للتتبع جدول lemma صغيرًا ونسخة احتياطية.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

الحالة الأخيرة هي لحظة التدريس الرئيسية. `watched` غير موجود في طاولتنا والبديل لدينا يعالج `ing` فقط. يغطي اللفظ الحقيقي `ed`، الأفعال الشاذة، الصفات المقارنة، صيغ الجمع مع تغيرات الصوت (`children -> child`). ولهذا السبب تستخدم أنظمة الإنتاج WordNet، أو مُورفولوجيزر spaCy، أو مُحلل مورفولوجي كامل.

### Step 4: pipe them together

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

القطعة المفقودة هي علامة POS. المرحلة 5 · 07 (POS وضع العلامات) تبني واحدة. في الوقت الحالي، قم بتعيين كل شيء افتراضيًا على `NOUN` والاعتراف بالقيد.

## Use It

NLTK وspaCy يشحنان إصدارات الإنتاج. بضعة أسطر لكل منهما.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` يتعامل مع الاختصارات وUnicode وحالات الحافة التي يفتقدها التعبير العادي الخاص بك. `PorterStemmer` يجري جميع المراحل الخمس. `WordNetLemmatizer` يحتاج إلى العلامة POS المترجمة من مخطط Penn Treebank الخاص بـ NLTK إلى مجموعة اختصارات WordNet. أسلاك الترجمة أعلاه هي الجزء الذي تتخطاه معظم البرامج التعليمية.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

يخفي spaCy خط pipe بالكامل خلف `nlp(text)`. يتم تشغيل كل من الترميز ووضع العلامات POS والتحويل. أسرع من NLTK على نطاق واسع. أكثر دقة من خارج منطقة الجزاء. المقايضة هي أنه لا يمكنك بسهولة تبديل المكونات الفردية.

### When to pick which

| الوضع | اختر |
|-----------|------|
| التدريس والبحث وتبادل المكونات | NLTK |
| الإنتاج، متعدد اللغات، السرعة مهمة | سباسي |
| المحول pipeline (سوف يتم ترميزه باستخدام رمز النموذج على أي حال) | استخدم `tokenizers` / `transformers` وتخطي المعالجة المسبقة الكلاسيكية |

### The two failure modes nobody warns you about

تقوم معظم البرامج التعليمية بتعليم الخوارزميات والتوقف. شيئان سوف يعضان المعالجة المسبقة الحقيقية pipeline، ولا يتم تغطيتهما أبدًا.

**انجراف قابلية التكرار.** NLTK وspaCy يغيران سلوك الرمز المميز وسلوك lemmatizer بين الإصدارات. ما أنتج `['do', "n't"]` في spaCy 2.x قد ينتج `["don't"]` في 3.x. تم تدريب النموذج الخاص بك على توزيع واحد. الاستدلال يعمل الآن على واحد مختلف. تتدهور الدقة بهدوء ولا أحد يعرف السبب. تثبيت إصدارات المكتبة في `requirements.txt`. اكتب اختبار انحدار المعالجة المسبقة الذي يجمد الترميز المتوقع لـ 20 عينة من الجمل. تشغيله على كل ترقية.

**عدم تطابق التدريب/الاستدلال.** التدريب باستخدام المعالجة المسبقة القوية (الأحرف الصغيرة، وإزالة كلمة التوقف، والحذف)، والنشر على مدخلات المستخدم الأولية، ومشاهدة فجوة الأداء. هذا هو فشل الإنتاج الأكثر شيوعًا NLP. إذا قمت بالمعالجة المسبقة أثناء التدريب، فيجب عليك تشغيل الوظيفة المماثلة أثناء الاستدلال. المعالجة المسبقة للسفينة كوظيفة داخل حزمة النموذج، وليس كخلية دفتر ملاحظات يعيد فريق الخدمة كتابتها.

## Ship It

مطالبة قابلة لإعادة الاستخدام تساعد المهندسين على اختيار استراتيجية المعالجة المسبقة دون قراءة ثلاثة كتب مدرسية.

حفظ باسم `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Exercises

1. **سهل.** قم بتمديد `tokenize` للاحتفاظ بعناوين URL كرموز مميزة واحدة. الاختبار: `tokenize("Visit https://example.com today.")` يجب أن ينتج رمزًا مميزًا URL واحد.
2. **متوسطة.** قم بتنفيذ خطوة بورتر 1ب. إذا كانت الكلمة تحتوي على حرف متحرك وتنتهي بـ `ed` أو `ing`، قم بإزالتها. تعامل مع قاعدة الحروف الساكنة المزدوجة (`hopping -> hop`، وليس `hopp`).
3. **صعب.** أنشئ أداة lemmatizer تستخدم WordNet كجدول بحث ولكنها ترجع إلى جهاز Porter الخاص بك عندما لا يكون هناك إدخال في WordNet. قياس الدقة في مجموعة العلامات مقابل WordNet العادي وPorter العادي.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| الرمز المميز | كلمة | أي وحدة يستهلكها النموذج. يمكن أن تكون كلمة أو كلمة فرعية أو حرف أو بايت. |
| الجذعية | جذر كلمة | نتيجة تجريد اللاحقة المستندة إلى القواعد. ليست دائما كلمة حقيقية. |
| ليما | نموذج القاموس | النموذج الذي ستبحث عنه. يتطلب السياق النحوي لحساب بشكل صحيح. |
| POS وسم | جزء من الكلام | فئة مثل NOUN، VERB، ADJ. هناك حاجة إلى lemmatize بدقة. |
| مورفولوجيا | قواعد شكل الكلمة | كيف يتغير شكل الكلمة بناءً على الزمن والرقم والحالة. يعتمد Lemmatization على ذلك. |

## Further Reading

- [بورتر، م. ف. (1980). خوارزمية لتجريد اللاحقة](https://tartarus.org/martin/PorterStemmer/def.txt) — الورقة الأصلية، خمس صفحات، لا تزال أوضح تفسير.
- [spaCy 101 — السمات اللغوية](https://spacy.io/usage/linguistic-features) — كيف يتم توصيل الخط pipe الحقيقي.
- [NLTK كتاب، الفصل 3](https://www.nltk.org/book/ch03.html) — حالات حافة الترميز التي لم تفكر بها بعد.
