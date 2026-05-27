# Bag of Words, TF-IDF, and Text Representation

> عد أولا، فكر لاحقا. TF-IDF لا يزال يتفوق على عمليات التضمين في المهام المحددة جيدًا في عام 2026.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 01 (معالجة النصوص)، المرحلة 2 · 02 (الانحدار الخطي من الصفر)
**الوقت:** ~75 دقيقة

## The Problem

النموذج يحتاج إلى أرقام. لديك سلاسل.

كل NLP pipeline يجب أن يجيب على نفس السؤال. كيف يمكننا تحويل تيار متغير الطول من الرموز المميزة إلى متجه ذي حجم ثابت يمكن أن يستهلكه المصنف. كانت الإجابة الأولى التي وصل إليها الميدان هي أغبى إجابة ناجحة. عد الكلمات. اصنع ناقل.

لقد حمل هذا المتجه إنتاجًا أكبر NLP من أي نموذج تضمين. مرشحات البريد العشوائي، ومصنفات المواضيع، واكتشاف شذوذ السجل، وترتيب البحث (قبل BM25)، والموجة الأولى من تحليل المشاعر، والعقد الأول من المعايير الأكاديمية NLP. لا يزال الممارسون في عام 2026 يصلون إليها أولاً في مهام التصنيف الضيقة. إنه سريع وقابل للتفسير، وغالبًا ما لا يمكن تمييزه عن نموذج التضمين ذي المعلمة 400 ميجا في المهام التي يكون فيها وجود الكلمة هو ما يهم.

يقوم هذا الدرس ببناء حقيبة من الكلمات، ثم TF-IDF، من الصفر. ثم يظهر scikit-learn وهو يفعل الشيء نفسه في ثلاثة أسطر. ثم قم بتسمية وضع الفشل الذي تصل إليه make للتضمين.

## The Concept

**حقيبة الكلمات (BoW)** ترمي النظام بعيدًا. لكل وثيقة، قم بحساب عدد المرات التي تظهر فيها كل كلمة من المفردات. طول المتجه هو حجم المفردات. الموضع `i` هو عدد الكلمات `i`.

**TF-IDF** يعيد وزن القوس. الكلمة التي تظهر في كل مستند غير مفيدة، لذا قم بتقليص حجمها. الكلمة النادرة في المجموعة ولكنها متكررة في مستند واحد هي الإشارة، لذا قم بتوسيع نطاقها.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

حيث `TF` هو تكرار المصطلح في المستند، `df` هو تكرار المستند (كم عدد المستندات التي تحتوي على الكلمة)، `N` هو إجمالي المستندات. يحافظ الرقم `log` على ثقل الكلمات الموجودة في كل مكان.

الخاصية الأساسية: كلاهما ينتجان متجهات متفرقة ذات محاور قابلة للتفسير. يمكنك إلقاء نظرة على أوزان المصنف المدرب وقراءة الكلمات التي تدفع المستند نحو كل فئة. لا يمكنك القيام بذلك باستخدام تضمين 768 بُعدًا BERT.

## Build It

### Step 1: build the vocabulary

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

الإدخال: قائمة بالمستندات التي تم ترميزها (أي أداة ترميز على مستوى الكلمة ستفي بالغرض؛ يستخدم `code/main.py` في هذا الدرس متغيرًا صغيرًا مبسطًا). الإخراج: `{word: index}` dict. ترتيب الإدراج الثابت يعني أن فهرس الكلمات 0 هو أول كلمة تظهر في المستند الأول. تختلف الاتفاقية؛ scikit-learn فرز أبجديا.

### Step 2: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

الصفوف هي المستندات. الأعمدة هي مؤشرات المفردات. الإدخال `[i][j]` هو "عدد المرات التي تظهر فيها الكلمة `j` في المستند `i`." يحتوي المستند 1 على `cat` مرتين لأنه فعل ذلك. يحتوي المستند 0 على `ran` صفر مرة لأنه لم يحدث ذلك.

### Step 3: term frequency and document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

حيلتان للتنعيم تستحقان التسمية. يتجنب `(n+1)/(d+1)` `log(x/0)`. تضمن القيمة الزائدة `+1` أن الكلمة في كل مستند لا تزال تحتوي على IDF 1 (وليس 0)، مطابقة للقيمة scikit-learn الافتراضية. تستخدم التطبيقات الأخرى الخام `log(N/df)`. كلاهما يعمل؛ النسخة السلسة أكثر ودا.

### Step 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

ثلاث وثائق، خمس كلمات مفردة (`the`، `cat`، `sat`، `dog`، `ran`). يظهر `the` في الثلاثة، لذا فإن IDF منخفض. يظهر `dog` في واحد، لذا فإن IDF مرتفع. المتجهات متفرقة (معظم الإدخالات صغيرة) والكلمات التمييزية تظهر.

### Step 5: L2-normalize rows

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

بدون التطبيع، تحصل الوثيقة الأطول على ناقل أكبر وتهيمن على درجات التشابه. L2 التطبيع يضع كل مستند على مساحة الوحدة الفائقة. أصبح تشابه جيب التمام بين الصفوف الآن مجرد منتج نقطي.

## Use It

scikit-learn يشحن نسخة الإنتاج.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` يقوم بالترميز والمفردات وBoW في مكالمة واحدة. `TfidfVectorizer` يضيف IDF الترجيح و L2 التطبيع. كلاهما يُرجعان مصفوفات متفرقة. بالنسبة إلى المستندات التي يصل حجمها إلى 100 ألف، لا تتناسب النسخة الكثيفة مع الذاكرة؛ البقاء متناثر حتى يتطلب المصنف كثيفة.

المقابض التي تغير كل شيء:

| أرج | تأثير |
|-----|--------|
| `ngram_range=(1, 2)` | قم بتضمين الصور الكبيرة. عادة ما يعزز التصنيف. |
| `min_df=2` | قم بإسقاط الكلمات في أقل من مستندين. التشذيب المفردات على البيانات الصاخبة. |
| `max_df=0.95` | أسقط الكلمات في أكثر من 95% من المستندات. يقترب من إزالة كلمة التوقف بدون قائمة مضمنة. |
| `stop_words="english"` | قائمة كلمات التوقف المدمجة في scikit-learn. يعتمد على المهمة - تحليل المشاعر يجب ألا يسقط النفي. |
| `sublinear_tf=True` | استخدم `1 + log(tf)` بدلاً من `tf` الخام. يساعد عندما يتكرر المصطلح عدة مرات في مستند واحد. |

### When TF-IDF still wins (as of 2026)

- اكتشاف البريد العشوائي، ووضع العلامات على المواضيع، ووضع علامة على شذوذ السجل. حضور الكلمة هو ما يهم؛ فارق بسيط الدلالية لا.
- الأنظمة منخفضة البيانات (مئات الأمثلة المصنفة). TF-IDF بالإضافة إلى الانحدار اللوجستي ليس له تكلفة تدريب مسبق.
- الكمون في أي مكان مهم. TF-IDF بالإضافة إلى نموذج خطي يجيب بالميكروثانية. يستغرق تضمين مستند من خلال محول ما بين 10 إلى 100 مللي ثانية.
- الأنظمة التي يجب أن تشرح توقعاتها. فحص معاملات المصنف. أهم الكلمات الإيجابية هي السبب.

### When TF-IDF fails

فشل العمى الدلالي. النظر في هاتين الوثيقتين:

- "الفيلم لم يكن جيداً على الإطلاق."
- "الفيلم كان ممتازا."

واحد هو مراجعة سلبية. واحد إيجابي. تداخلهم TF-IDF هو بالضبط `{the, movie, was}`. يجب أن يتذكر مُصنف حقيبة الكلمات أن الكلمة `not` بالقرب من `good` تقلب التسمية. ويمكنه تعلم ذلك من خلال بيانات كافية، ولكن ليس بنفس القدر من الرشاقة التي يتمتع بها النموذج الذي يفهم بناء الجملة.

الفشل الآخر: الكلمات خارج المفردات عند الاستدلال. ليس لدى نموذج BoW الذي تم تدريبه على مراجعات IMDb أي فكرة عما يجب فعله بـ `Zoomer-approved` إذا لم يظهر هذا الرمز المميز في التدريب مطلقًا. تعالج تضمينات الكلمات الفرعية (الدرس 04) هذا الأمر. TF-IDF لا يمكن.

### Hybrid: TF-IDF weighted embeddings

الافتراضي العملي لعام 2026 لتصنيف البيانات المتوسطة: استخدم أوزان TF-IDF للانتباه إلى تضمينات الكلمات.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

يمكنك الحصول على القدرة الدلالية من التضمينات والتركيز على الكلمات النادرة من TF-IDF. يتدرب المصنف على المتجهات المجمعة. يتفوق هذا في الأداء من تلقاء نفسه فيما يتعلق بتصنيف المشاعر والموضوع والنية أدناه حوالي 50 ألف من الأمثلة المصنفة.

## Ship It

حفظ باسم `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Exercises

1. **سهل.** قم بتنفيذ `cosine_similarity(doc_vec_a, doc_vec_b)` على إخراج L2-تطبيع TF-IDF. تحقق من أن المستندات المتطابقة تحصل على 1.0 وأن المستندات المنفصلة ذات المفردات تحصل على 0.0.
2. **متوسط.** أضف دعم `n-gram` إلى `bag_of_words`. تنتج المعلمة `n` أعدادًا تزيد عن `n` جرامات. اختبر أن `n=2` على `["the", "cat", "sat"]` ينتج عنها أعداد بيجرام لـ `["the cat", "cat sat"]`.
3. **صعب.** قم ببناء الهجين التضمين الموزون TF-IDF أعلاه باستخدام ناقلات GloVe 100d (التنزيل مرة واحدة، ذاكرة التخزين المؤقت). قارن دقة التصنيف مع التضمينات العادية TF-IDF والتضمينات المجمعة المتوسطة في مجموعة بيانات 20 مجموعة أخبار. تقرير الذي يفوز أين.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| القوس | ناقل تردد الكلمة | عدد الكلمات من المفردات في وثيقة واحدة. يرمي بعيدا النظام. |
| TF | تردد المصطلح | عدد الكلمات في المستند، ويتم ضبطه اختياريًا حسب طول المستند. |
| DF | تردد الوثيقة | عدد المستندات التي تحتوي على الكلمة مرة واحدة على الأقل. |
| IDF | تردد الوثيقة العكسية | `log(N / df)` ناعم. يقلل من وزن الكلمات التي تظهر في كل مكان. |
| ناقل متفرق | في الغالب أصفار | المفردات عادة ما تكون من 10 آلاف إلى 100 ألف كلمة؛ معظمهم غائبون عن أي وثيقة معينة. |
| تشابه جيب التمام | زاوية المتجهات | المنتج النقطي للمتجهات المقيسة L2. 1 متطابق، 0 متعامد. |

## Further Reading

- [scikit — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — the canonical API reference, plus notes on every knob.
- [Salton, G., & Buckley, C. (1988). أساليب ترجيح المصطلح في استرجاع النص تلقائيًا](https://www.sciencedirect.com/science/article/pii/0306457388900210) — الورقة التي جعلت TF-IDF الورقة الافتراضية لمدة عقد من الزمن.
- ["لماذا TF-IDF لا تزال تتفوق على التضمينات" - Ashfaque Thonikkadavan (متوسط)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) - 2026) عندما تفوز الطريقة القديمة ولماذا.
