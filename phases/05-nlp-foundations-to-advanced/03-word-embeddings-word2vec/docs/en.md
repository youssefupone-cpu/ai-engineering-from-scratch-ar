# Word Embeddings — Word2Vec from Scratch

> الكلمة هي الشركة التي تحافظ عليها. قم بتدريب شبكة ضحلة على هذه الفكرة وستسقط الهندسة.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 5 · 02 (BoW + TF-IDF)، المرحلة 3 · 03 (الانتشار العكسي من الصفر)
**الوقت:** ~75 دقيقة

## The Problem

TF-IDF يعرف أن `dog` و `puppy` كلمتان مختلفتان. ولا يعلم أنهما يقصدان نفس الشيء تقريبًا. لا يمكن للمصنف الذي تم تدريبه على `dog` تعميم مراجعة حول `puppy`. يمكنك تجاوز هذا عن طريق إدراج المرادفات، لكن هذا يفشل في المصطلحات النادرة، ومصطلحات المجال، وكل لغة لم تتوقعها.

تريد تمثيلًا حيث `dog` و `puppy` يهبطان بالقرب من بعضهما البعض في الفضاء. حيث `king - man + woman` تقع بالقرب من `queen`. حيث يقوم نموذج تم تدريبه على `dog` بنقل بعض الإشارات إلى `puppy` مجانًا.

لقد أعطانا Word2Vec هذه المساحة. شبكة عصبية مكونة من طبقتين، عمليات تدريب تريليون رمز، تم نشرها في عام 2013. الهندسة المعمارية بسيطة إلى حد محرج تقريبًا. تم إعادة تشكيل النتائج NLP لمدة عقد من الزمن.

## The Concept

**فرضية التوزيع** (فيرث، 1957): "سوف تعرف الكلمة من خلال الشركة التي تحتفظ بها." إذا ظهرت كلمتان في سياقات مماثلة، فمن المحتمل أنهما يعنيان أشياء مماثلة.

يأتي Word2Vec في نسختين، وكلاهما يستغل هذه الفكرة.

- **Skip-gram.** بالنظر إلى الكلمة المركزية، توقع الكلمات المحيطة. `cat -> (the, sat, on)` بحجم النافذة 2.
- **CBOW (حقيبة الكلمات المستمرة).** بالنظر إلى الكلمات المحيطة، توقع المركز. `(the, sat, on) -> cat`.

يعد Skip-gram أبطأ في التدريب ولكنه يتعامل مع الكلمات النادرة بشكل أفضل. أصبح الافتراضي.

تحتوي الشبكة على طبقة مخفية واحدة بدون خطية. الإدخال هو ناقل واحد ساخن على المفردات. الإخراج هو softmax على المفردات. بعد التدريب، يمكنك التخلص من طبقة الإخراج. أوزان الطبقة المخفية هي التضمينات.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

الحيلة: softmax أكثر من 100 ألف كلمة باهظ الثمن. يستخدم Word2Vec ** أخذ العينات السلبية ** لتحويله إلى مهمة تصنيف ثنائية. توقع "هل ظهرت كلمة السياق هذه بالقرب من الكلمة المركزية، نعم أم لا". قم بتجربة مجموعة من الكلمات السلبية (غير المتزامنة) لكل زوج تدريب بدلاً من حساب softmax على المفردات بأكملها.

## Build It

### Step 1: training pairs from a corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

يعد كل زوج (مركز، سياق) في النافذة مثالًا تدريبيًا إيجابيًا.

### Step 2: embedding tables

مصفوفتان. `W` هو جدول تضمين الكلمات المركزية (الجدول الذي تحتفظ به). `W'` هو جدول الكلمات السياقية (غالبًا ما يتم تجاهله، وأحيانًا يتم حساب متوسطه بـ `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

حرف عشوائي صغير. حجم المفردة 10k وخافت 100 واقعي؛ للتدريس، 50 حرفًا × 16 خافتًا كافية لرؤية الهندسة.

### Step 3: negative sampling objective

لكل زوج موجب `(center, context)`، عينة `k` كلمات عشوائية من المفردات ككلمات سلبية. قم بتدريب النموذج بحيث يكون حاصل الضرب النقطي `W[center] · W'[context]` عاليًا للإيجابيات ومنخفضًا للسالبة.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

الصيغة السحرية: الخسارة اللوجستية على الزوج الموجب (تريد السيني بالقرب من 1) بالإضافة إلى الخسارة اللوجستية على الأزواج السالبة (تريد السيني بالقرب من 0). تتدفق التدرجات إلى كلا الجدولين. الاشتقاق الكامل موجود في الورقة الأصلية؛ قم بالتمرير من خلاله مرة واحدة باستخدام قلم رصاص وورقة إذا كنت تريد أن تلتصق.

### Step 4: train on a toy corpus

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

بعد فترة كافية من مجموعة كبيرة، الكلمات التي تشترك في السياقات لها تضمينات مركزية مماثلة. على مجموعة الألعاب، ترى التأثير بشكل ضعيف. على مليارات العملات، ترى ذلك بشكل كبير.

### Step 5: the analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

في ناقلات أخبار Google 300d المدربة مسبقًا:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. ليس لأن العارضة تعرف ما هي الملوكية. لأن المتجه `(king - man)` يلتقط شيئًا مثل "ملكي"، ويضيفه إلى الأراضي `woman` القريبة من منطقة الأنثى الملكية.

## Use It

كتابة Word2Vec من الصفر هي عملية تعليمية. يستخدم الإنتاج NLP `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

بالنسبة للعمل الحقيقي، فإنك تقريبًا لا تقوم بتدريب Word2Vec بنفسك. يمكنك تنزيل المتجهات المدربة مسبقًا.

- **GloVe** — نهج تحليل مصفوفة التواجد المشترك إلى عوامل ستانفورد. نقاط التفتيش 50د، 100د، 200د، 300د. تغطية عامة جيدة. يغطي الدرس 04 GloVe على وجه التحديد.
- **fastText** — ملحق Word2Vec الخاص بفيسبوك والذي يتضمن حرف n-gram. يعالج الكلمات خارج المفردات عن طريق تكوين كلمات فرعية. الدرس 04.
- ** Word2Vec المدرب مسبقًا على أخبار Google ** - 300d، مفردات 3M كلمة، تم نشرها عام 2013. ولا يزال يتم تنزيلها يوميًا.

### When Word2Vec still wins in 2026

- استرجاع خفيف الوزن خاص بالمجال. تدرب على الملخصات الطبية في ساعة واحدة على جهاز كمبيوتر محمول، واحصل على ناقلات متخصصة دون التقاط نموذج عام.
- هندسة الميزات بأسلوب التناظر. `gender_vector = mean(man - woman pairs)`. اطرحها من الكلمات الأخرى للحصول على محور محايد جنسانيًا. لا تزال تستخدم في أبحاث العدالة.
- القابلية للتفسير. 100d صغير بما يكفي للتخطيط عبر PCA أو t-SNE ورؤية شكل المجموعات فعليًا.
- يجب تشغيل الاستدلال في أي مكان على الجهاز بدون GPU. بحث Word2Vec هو جلب صف واحد.

### Where Word2Vec fails

جدار تعدد المعاني. `bank` لديه متجه واحد. `river bank` و `financial bank` شاركها. `table` (جدول البيانات مقابل الأثاث) يشاركه. لا يمكن للمصنف في اتجاه مجرى النهر أن يميز الحواس عن المتجه.

التضمين السياقي (ELMo، BERT، كل محول منذ ذلك الحين) حل هذه المشكلة عن طريق إنتاج متجه مختلف لكل تكرار للكلمة بناءً على السياق المحيط. هذه هي القفزة من Word2Vec إلى BERT: من الثابت إلى السياقي. المرحلة 7 تغطي نصف المحول.

مشكلة الخروج من المفردات هي الفشل الآخر. لم يشاهد Word2Vec أبدًا `Zoomer-approved` إذا لم يكن موجودًا في بيانات التدريب. لا يوجد احتياطي. يقوم fastText بإصلاح هذه المشكلة من خلال تكوين الكلمات الفرعية (الدرس 04).

## Ship It

حفظ باسم `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Exercises

1. **سهل.** قم بتشغيل حلقة التدريب على مجموعة صغيرة (20 جملة عن القطط والكلاب). بعد 200 حقبة، تحقق من أن `nearest(vocab, W, W[vocab["cat"]])` ترجع `dog` في أعلى 3. إذا لم يكن الأمر كذلك، قم بزيادة العصور أو المفردات.
2. **متوسطة.** أضف عينات فرعية من الكلمات المتكررة. يتم حذف الكلمات التي يزيد تكرارها عن `10^-5` من أزواج التدريب باحتمال يتناسب مع تكرارها. قياس التأثير على تشابه الكلمات النادرة.
3. **صعب.** تدريب نموذج على مجموعة الأخبار العشرين. حساب محوري التحيز: `he - she` و `doctor - nurse`. مشروع كلمات المهنة على كلا المحورين. قم بالإبلاغ عن المهن التي بها أكبر فجوة تحيز. هذا هو نوع التحقيق الذي يستخدمه الباحثون في العدالة.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| تضمين الكلمات | كلمة كمتجه | تمثيل كثيف ومنخفض التعتيم (عادةً 100-300) تم تعلمه من السياق. |
| تخطي جرام | خدعة Word2Vec | توقع كلمات السياق من الكلمة المركزية. أبطأ من CBOW، أفضل للكلمات النادرة. |
| أخذ العينات السلبية | اختصار التدريب | استبدل softmax فوق المفردات الكاملة بالتصنيف الثنائي مقابل `k` كلمات عشوائية. |
| التضمين الثابت | ناقل واحد لكل كلمة | نفس المتجهات بغض النظر عن السياق. فشل في تعدد المعاني. |
| التضمين السياقي | ناقلات حساسة للسياق | ناقلات مختلفة لكل حدث بناءً على الكلمات المحيطة. ما تنتجه المحولات. |
| OOV | خارج المفردات | كلمة لم تظهر في التدريب. لا يمكن لـ Word2Vec إنتاج متجه لهذه العناصر. |

## Further Reading

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — the negative-sampling paper. Short and readable.
- [Rong, X. (2014). شرح تعلم معلمة word2vec](https://arxiv.org/abs/1411.2738) — أوضح اشتقاق للتدرجات، إذا كانت الرياضيات في الورقة الأصلية تبدو كثيفة.
- [البرنامج التعليمي gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) — إعدادات التدريب على الإنتاج التي تعمل بالفعل.
