# POS Tagging and Syntactic Parsing

> كانت القواعد غير عصرية لفترة من الوقت. ثم كل LLM pipخط مطلوب للتحقق من صحة الاستخراج المنظم، وقد عاد.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 5 · 01 (معالجة النصوص)، المرحلة 2 · 14 (نايف بايز)
**الوقت:** ~45 دقيقة

## The Problem

وعد الدرس 01 بأن عملية اللفظ تحتاج إلى علامة جزء من الكلام. بدون معرفة أن `running` هو فعل، لا يمكن للمترجم أن يختزله إلى `run`. بدون معرفة أن `better` صفة، لا يمكن اختزالها إلى `good`.

لقد أخفى هذا الوعد مجالًا فرعيًا كاملاً. تقوم علامات جزء من الكلام بتعيين فئات نحوية. يستعيد التحليل النحوي بنية شجرة الجملة: أي كلمة تعدل أي فعل، وأي فعل يحكم أي وسيطات. الكلاسيكية NLP قضت عشرين عامًا في تحسين كليهما. ثم قام التعلم العميق بدمجها في مهمة تصنيف رمزية فوق محول مُدرب مسبقًا، ومضى مجتمع البحث قدمًا.

وليس المجتمع التطبيقي. لا يزال كل استخراج منظم pipeline يستخدم POS وأشجار التبعية تحت الغطاء. LLM-تم إنشاء JSON يتم التحقق من صحته مقابل القيود النحوية. تعمل أنظمة الإجابة على الأسئلة على تحليل الاستعلامات باستخدام تحليلات التبعية. يتحقق مقيمو جودة الترجمة الآلية من محاذاة أشجار التحليل.

يستحق المعرفة. يقدم هذا الدرس مجموعات العلامات والخطوط الأساسية والنقطة التي تتوقف عندها عن التنفيذ من البداية وتستدعي spaCy.

## The Concept

**POS وضع العلامات** يقوم بتسمية كل رمز مميز بفئة نحوية. مجموعة العلامات **Penn Treebank (PTB)** هي العلامة الافتراضية باللغة الإنجليزية. 36 علامة مع الفروق التي يجدها القارئ العادي مثيرة للاهتمام: `NN` اسم المفرد، `NNS` اسم الجمع، `NNP` اسم العلم المفرد، `VBD` الفعل الماضي، `VBZ` الفعل ضمير الغائب المفرد، وما إلى ذلك. مجموعة العلامات **التبعيات العالمية (UD)** أكثر خشونة (17 علامة) ولا تعرف اللغة؛ أصبح الإعداد الافتراضي للعمل متعدد اللغات.

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**التحليل النحوي** ينتج عنه شجرة. نمطين رئيسيين:

- **تحليل الدوائر الانتخابية.** تتداخل العبارات الاسمية وعبارات الفعل وعبارات حروف الجر داخل بعضها البعض. الإخراج عبارة عن شجرة من الفئات غير الطرفية (NP، VP، PP) مع الكلمات كأوراق.
- **تحليل التبعية.** تحتوي كل كلمة على كلمة رئيسية واحدة تعتمد عليها، ومُصنفة بعلاقة نحوية. الإخراج عبارة عن شجرة حيث كل حافة هي (رأس، تابع، علاقة) ثلاثية.

فاز تحليل التبعية في العقد الأول من القرن الحادي والعشرين لأنه يعمم بشكل واضح عبر اللغات، وخاصة اللغات ذات الترتيب الحر للكلمات.

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## Build It

### Step 1: most-frequent-tag baseline

أغبى علامة POS فعالة. لكل كلمة، توقع العلامة التي كانت موجودة في أغلب الأحيان في التدريب.

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

في المجموعة البنية، تصل دقة خط الأساس هذا إلى 85% تقريبًا. ليست جيدة، ولكن الأرضية التي لا ينبغي أن يسقط تحتها أي نموذج جدي.

### Step 2: bigram HMM tagger

نموذج الاحتمال المشترك للتسلسل:

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

جدولان: احتمالات الانتقال (العلامة المعطاة للعلامة السابقة)، احتمالات الانبعاث (الكلمة المعطاة للعلامة). قم بتقدير كليهما من خلال التعدادات باستخدام تجانس لابلاس. فك التشفير باستخدام Viterbi (البرمجة الديناميكية عبر شبكة العلامة).

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

بيجرام HMM على براون يصل إلى دقة تصل إلى 93٪. القفزة من 85% إلى 93% هي في الغالب احتمالات انتقالية - يتعلم النموذج أن `DET NOUN` أمر شائع وأن `NOUN DET` نادر.

### Step 3: why modern taggers beat this

الانتقال + احتمالات الانبعاثات محلية. لا يمكنهم إدراك أن `saw` هو اسم في عبارة "اشتريت منشارًا" ولكنه فعل في "لقد شاهدت الفيلم". A CRF بميزات عشوائية (اللاحقة، شكل الكلمة، الكلمة قبلها وبعدها، الكلمة نفسها) تصل إلى 97% تقريبًا. A BiLSTM-CRF أو محول يصل إلى ~98%+.

يتم تحديد الحد الأقصى لهذه المهمة من خلال عدم توافق الحواشي. يتفق المدونون البشريون على ما يقرب من 97% من الوقت في Penn Treebank. من المحتمل أن تكون النماذج التي تجاوزت 98٪ قد تجاوزت مجموعة الاختبار.

### Step 4: dependency parsing sketch

تحليل التبعية الكاملة من البداية خارج النطاق؛ معالجة الكتب المدرسية الأساسية موجودة في جورافسكي ومارتن. عائلتين كلاسيكيتين يجب معرفتهما:

- يعمل المحللون اللغويون **المعتمدون على الانتقال** (arc-eager, arc-standard) كمحلل لتقليل الإزاحة: فهم يقرأون الرموز المميزة، وينقلونها إلى مكدس، ويطبقون إجراءات التصغير التي تنشئ أقواسًا. فك التشفير الجشع سريع. التنفيذ الكلاسيكي هو MaltParser. النسخة العصبية الحديثة: المحلل اللغوي القائم على الانتقال لتشن ومانينغ.
- يقوم المحللون اللغويون **المعتمدون على الرسم البياني** (خوارزمية آيسنر، Dozat-Manning biaffine) بتسجيل كل حافة محتملة تعتمد على الرأس واختيار الحد الأقصى للشجرة الممتدة. أبطأ ولكن أكثر دقة.

بالنسبة لمعظم الأعمال التطبيقية، اتصل بـ SpaCy:

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

اقرأ العمود `dep` من الأسفل إلى الأعلى وستسقط البنية النحوية للجملة.

## Use It

يتم شحن كل مكتبة إنتاج NLP POS وموزعي التبعية كجزء من خط pipeline القياسي.

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`). سريع ودقيق ومتكامل مع الترميز + NER + lemmatization. `token.tag_` (بنسلفانيا)، `token.pos_` (UD)، `token.dep_` (علاقة التبعية).
- **ستانفورد NLP (مقطع)**. خليفة ستانفورد لـ CoreNLP. أحدث ما توصلت إليه التكنولوجيا في أكثر من 60 لغة.
- **ترانكيت**. يعتمد على المحولات، ودقة UD جيدة.
- **NLTK**. `pos_tag`. صالحة للاستعمال، بطيئة، كبار السن. غرامة للتدريس.

### Where this still matters in 2026

- **الترجمة.** يحتاج الدرس 01 إلى POS للترجمة بشكل صحيح. دائماً.
- **الاستخراج المنظم من مخرجات LLM.** التحقق من أن الجملة التي تم إنشاؤها تحترم القيود النحوية (على سبيل المثال، اتفاق الموضوع والفعل، المعدلات المطلوبة).
- **المشاعر القائمة على الجانب.** يخبرك تحليل التبعية بالصفة التي تعدل أي اسم.
- **فهم الاستعلام.** تتحلل "الأفلام التي أخرجها ويس أندرسون وبطولة بيل موراي" إلى قيود منظمة من خلال التحليل.
- **النقل عبر اللغات.** UD العلامات وعلاقات التبعية لا تعرف اللغة، مما يتيح التحليل المنظم للغات الجديدة.
- **حساب منخفض pipelines.** إذا لم تتمكن من شحن محول، POS + تحليل التبعية + المعجم الجغرافي يجعلك بعيدًا بشكل مدهش.

## Ship It

حفظ باسم `outputs/skill-grammar-pipelineeline.md`:

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## Exercises

1. **سهل.** باستخدام الخط الأساسي للعلامات الأكثر تكرارًا في مجموعة صغيرة من العلامات (على سبيل المثال، مجموعة براون الفرعية NLTK)، قم بقياس الدقة في الجمل المعلقة. التحقق من النتيجة ~ 85%.
2. **متوسط.** قم بتدريب البيجرام HMM أعلاه وقم بالإبلاغ عن دقة/استدعاء كل علامة. ما هي العلامات التي يربكها HMM أكثر؟
3. **صعب.** استخدم تحليل التبعية الخاص بـ spaCy لاستخراج ثلاثية الفاعل والفعل والمفعول به من عينة مكونة من 1000 جملة. قم بتقييم 50 ثلاثية تم تصنيفها يدويًا. توثيق حيث يفشل الاستخراج (غالبًا ما يكون سلبيًا، وإحداثيات، ومواضيع محذوفة).

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| POS وسم | نوع الكلمة | الفئة النحوية. PTB لديه 36؛ UD فيه 17. |
| بن تريبانك | مجموعة العلامات القياسية | خاصة باللغة الإنجليزية. أزمنة الفعل الدقيقة ورقم الاسم. |
| التبعيات العالمية | علامات متعددة اللغات | أكثر خشونة من PTB؛ لغة محايدة؛ الافتراضيات للعمل عبر اللغات. |
| تحليل التبعية | شجرة الجملة | كل كلمة لها رأس واحد، ولكل حرف علاقة نحوية. |
| فيتربي | البرمجة الديناميكية | يبحث عن تسلسل العلامات ذو الاحتمالية الأعلى في ضوء الانبعاثات والانتقالات. |

## Further Reading

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — the canonical textbook treatment of POS and parsing.
- [Universal Dependencies project](https://universaldependencies.org/) — the cross-lingual tagset and treebank collection used by every multilingual parser.
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — practical reference for every attribute exposed on `Token`.
- [Chen and Manning (2014). محلل تبعية سريع ودقيق باستخدام الشبكات العصبية](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — الورقة التي جلبت المحللين العصبيين إلى الاتجاه السائد.
