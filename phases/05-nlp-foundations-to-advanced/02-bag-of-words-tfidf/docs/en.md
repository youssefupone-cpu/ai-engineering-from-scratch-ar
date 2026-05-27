# حقيبة الكلمات، TF-IDF، وتمثيل النص
> عد أولا، فكر لاحقا. TF-IDF لا يزال يتفوق على عمليات التضمين في المهام المحددة جيدًا في عام 2026.
**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 5 · 01 (معالجة النصوص)، المرحلة 2 · 02 (الانحدار الخطي من الصفر)
**الوقت:** ~75 دقيقة
## المشكلة
النموذج يحتاج إلى أرقام. لديك سلاسل.
يجب على كل NLP pipe أن يجيب على نفس السؤال. كيف يمكننا تحويل تيار متغير الطول من الرموز المميزة إلى متجه ذي حجم ثابت يمكن أن يستهلكه المصنف. كانت الإجابة الأولى التي وصل إليها الميدان هي أغبى إجابة ناجحة. عد الكلمات. اصنع ناقل.
لقد حمل هذا المتجه إنتاجًا أكبر NLP من أي نموذج تضمين. مرشحات البريد العشوائي، ومصنفات المواضيع، واكتشاف شذوذ السجل، وتصنيف البحث (قبل BM25)، والموجة الأولى من تحليل المشاعر، والعقد الأول من معايير NLP الأكاديمية. لا يزال الممارسون في عام 2026 يصلون إليها أولاً في مهام التصنيف الضيقة. إنه سريع وقابل للتفسير، وغالبًا ما لا يمكن تمييزه عن نموذج التضمين ذي المعلمة 400 ميجا في المهام التي يكون فيها وجود الكلمة هو ما يهم.
يقوم هذا الدرس ببناء حقيبة من الكلمات، ثم TF-IDF، من الصفر. ثم يظهر scikit-learn وهو يفعل الشيء نفسه في ثلاثة أسطر. ثم قم بتسمية وضع الفشل الذي تصل إليه make للتضمين.
##المفهوم
**حقيبة الكلمات (BoW)** ترمي النظام بعيدًا. لكل وثيقة، قم بحساب عدد المرات التي تظهر فيها كل كلمة من المفردات. طول المتجه هو حجم المفردات. الموضع `i` هو عدد الكلمات `i`.
**TF-IDF** إعادة وزن القوس. الكلمة التي تظهر في كل مستند غير مفيدة، لذا قم بتقليص حجمها. الكلمة النادرة في المجموعة ولكنها متكررة في مستند واحد هي الإشارة، لذا قم بتوسيع نطاقها.
```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

حيث `TF` هو تكرار المصطلح في المستند، `df` هو تكرار المستند (كم عدد المستندات التي تحتوي على الكلمة)، `N` هو إجمالي المستندات. يحافظ `log` على وزن الكلمات الموجودة في كل مكان.
الخاصية الأساسية: كلاهما ينتجان متجهات متفرقة ذات محاور قابلة للتفسير. يمكنك إلقاء نظرة على أوزان المصنف المدرب وقراءة الكلمات التي تدفع المستند نحو كل فئة. لا يمكنك القيام بذلك باستخدام تضمين BERT ذو أبعاد 768.
## بنائها
### الخطوة 1: بناء المفردات
```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

الإدخال: قائمة بالمستندات التي تم ترميزها (أي أداة ترميز على مستوى الكلمة ستفي بالغرض؛ يستخدم `code/main.py` في هذا الدرس متغيرًا صغيرًا مبسطًا). الإخراج: `{word: index}` ديك. ترتيب الإدراج الثابت يعني أن فهرس الكلمات 0 هو أول كلمة تظهر في المستند الأول. تختلف الاتفاقية؛ scikit-learn يتم الفرز أبجديًا.
### الخطوة الثانية: حقيبة الكلمات
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

الصفوف هي المستندات. الأعمدة هي مؤشرات المفردات. الإدخال `[i][j]` هو "عدد المرات التي تظهر فيها الكلمة `j` في المستند `i`." يحتوي المستند 1 على `cat` مرتين لأنه فعل ذلك. يحتوي المستند 0 على `ran` صفر مرة لأنه لم يفعل ذلك.
### الخطوة 3: تكرار المصطلح وتكرار الوثيقة
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

حيلتان للتنعيم تستحقان التسمية. يتجنب `(n+1)/(d+1)` `log(x/0)`. يضمن `+1` اللاحق أن الكلمة في كل مستند لا تزال تحتوي على IDF 1 (وليس 0)، وهو ما يطابق الافتراضي scikit-learn. تستخدم التطبيقات الأخرى `log(N/df)` الخام. كلاهما يعمل؛ النسخة السلسة أكثر ودا.
### الخطوة 4: TF-IDF
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

ثلاث مستندات، خمس كلمات مفردة (`the`، `cat`، `sat`، `dog`، `ran`). يظهر `the` في الثلاثة، لذا فإن IDF الخاص به منخفض. يظهر `dog` في واحد، لذا فإن IDF الخاص به مرتفع. المتجهات متفرقة (معظم الإدخالات صغيرة) والكلمات التمييزية تظهر.
### الخطوة 5: L2-تسوية الصفوف
```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

بدون التطبيع، تحصل الوثيقة الأطول على ناقل أكبر وتهيمن على درجات التشابه. L2 يضع التطبيع كل مستند في الكرة الفائقة للوحدة. أصبح تشابه جيب التمام بين الصفوف الآن مجرد منتج نقطي.
## استخدمه
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

`CountVectorizer` يقوم بالترميز والمفردات وBoW في مكالمة واحدة. يضيف `TfidfVectorizer` IDF الترجيح وL2 التسوية. كلاهما يُرجعان مصفوفات متفرقة. بالنسبة إلى المستندات التي يصل حجمها إلى 100 ألف، لا تتناسب النسخة الكثيفة مع الذاكرة؛ البقاء متناثر حتى يتطلب المصنف كثيفة.
المقابض التي تغير كل شيء:
| أرج | تأثير |
|-----|--------|
| `ngram_range=(1, 2)` | قم بتضمين الصور الكبيرة. عادة ما يعزز التصنيف. |
| __الكود_1__ | قم بإسقاط الكلمات في أقل من مستندين. التشذيب المفردات على البيانات الصاخبة. |
| __الكود_2__ | أسقط الكلمات في أكثر من 95% من المستندات. يقترب من إزالة كلمة التوقف بدون قائمة مضمنة. |
| __الكود_3__ | قائمة كلمات التوقف المدمجة في scikit-learn. يعتمد على المهمة - تحليل المشاعر يجب ألا يسقط النفي. |
| __الكود_4__ | استخدم `1 + log(tf)` بدلاً من `tf` الخام. يساعد عندما يتكرر المصطلح عدة مرات في مستند واحد. |
### عندما يستمر TF-IDF في الفوز (اعتبارًا من 2026)
- اكتشاف البريد العشوائي، ووضع العلامات على المواضيع، ووضع علامة على شذوذ السجل. حضور الكلمة هو ما يهم؛ فارق بسيط الدلالية لا.
- الأنظمة منخفضة البيانات (مئات الأمثلة المصنفة). TF-IDF بالإضافة إلى الانحدار اللوجستي ليس له تكلفة تدريب مسبق.
- الكمون في أي مكان مهم. TF-IDF بالإضافة إلى نموذج خطي يجيب بالميكروثانية. يستغرق تضمين مستند من خلال محول ما بين 10 إلى 100 مللي ثانية.
- الأنظمة التي يجب أن تشرح توقعاتها. فحص معاملات المصنف. أهم الكلمات الإيجابية هي السبب.
### عندما يفشل TF-IDF
فشل العمى الدلالي. النظر في هاتين الوثيقتين:
- "الفيلم لم يكن جيداً على الإطلاق."
- "الفيلم كان ممتازا."
واحد هو مراجعة سلبية. واحد إيجابي. تداخل TF-IDF هو بالضبط `{the, movie, was}`. يجب أن يتذكر مُصنف كيس الكلمات أن الكلمة `not` بالقرب من `good` تقلب التسمية. ويمكنه تعلم ذلك من خلال بيانات كافية، ولكن ليس بنفس القدر من الرشاقة التي يتمتع بها النموذج الذي يفهم بناء الجملة.
الفشل الآخر: الكلمات خارج المفردات عند الاستدلال. ليس لدى نموذج BoW الذي تم تدريبه على مراجعات IMDb أي فكرة عما يجب فعله مع `Zoomer-approved` إذا لم يظهر هذا الرمز المميز في التدريب مطلقًا. تعالج تضمينات الكلمات الفرعية (الدرس 04) هذا الأمر. TF-IDF لا يمكن.
### الهجين: TF-IDF التضمينات الموزونة
الإعداد الافتراضي العملي لعام 2026 لتصنيف البيانات المتوسطة: استخدم أوزان TF-IDF للانتباه إلى تضمينات الكلمات.
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

يمكنك الحصول على سعة دلالية من عمليات التضمين، والتركيز على الكلمات النادرة من TF-IDF. يتدرب المصنف على المتجهات المجمعة. يتفوق هذا في الأداء من تلقاء نفسه فيما يتعلق بتصنيف المشاعر والموضوع والنية أدناه حوالي 50 ألف من الأمثلة المصنفة.
## اشحنها
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

## تمارين
1. **سهل.** قم بتنفيذ `cosine_similarity(doc_vec_a, doc_vec_b)` على إخراج L2-المعدل TF-IDF. تحقق من أن المستندات المتطابقة تحصل على 1.0 وأن المستندات المنفصلة ذات المفردات تحصل على 0.0.
2. **متوسط.** أضف دعم `n-gram` إلى `bag_of_words`. تنتج المعلمة `n` أعدادًا تزيد عن `n` جرامًا. اختبر أن `n=2` على `["the", "cat", "sat"]` ينتج عنه أعداد بيجرام لـ `["the cat", "cat sat"]`.
3. **صعب.** أنشئ TF-IDF المختلط للتضمين الموزون أعلاه باستخدام متجهات GloVe 100d (تنزيل مرة واحدة، تخزين مؤقت). قارن دقة التصنيف مع TF-IDF والتضمينات المجمعة البسيطة في مجموعة بيانات 20 مجموعة أخبار. تقرير الذي يفوز أين.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| القوس | ناقل تردد الكلمة | عدد الكلمات من المفردات في وثيقة واحدة. يرمي بعيدا النظام. |
| TF | تردد المصطلح | عدد الكلمات في المستند، ويتم ضبطه اختياريًا حسب طول المستند. |
| __المصطلح_1__ | تردد الوثيقة | عدد المستندات التي تحتوي على الكلمة مرة واحدة على الأقل. |
| __المصطلح_2__ | تردد الوثيقة العكسية | `log(N / df)` ممهد. يقلل من وزن الكلمات التي تظهر في كل مكان. |
| ناقل متفرق | في الغالب أصفار | المفردات عادة ما تكون من 10 آلاف إلى 100 ألف كلمة؛ معظمهم غائبون عن أي وثيقة معينة. |
| تشابه جيب التمام | زاوية المتجهات | المنتج النقطي للمتجهات التي تمت تسويتها L2. 1 متطابق، 0 متعامد. |
## مزيد من القراءة
- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — المرجع الأساسي API، بالإضافة إلى ملاحظات على كل مقبض.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — الورقة التي جعلت TF-IDF الورقة الافتراضية لمدة عقد من الزمن.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — عام 2026 سيبدأ عندما تفوز الطريقة القديمة ولماذا.