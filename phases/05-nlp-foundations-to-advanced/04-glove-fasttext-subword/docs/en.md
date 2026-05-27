# GloVe, FastText, and Subword Embeddings

> قام Word2Vec بتدريب عملية تضمين واحدة لكل كلمة. قام GloVe بتحليل مصفوفة التواجد المشترك. قام FastText بدمج القطع. BPE موصول للمحولات.

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 5 · 03 (Word2Vec من الصفر)
**الوقت:** ~45 دقيقة

## The Problem

ترك Word2Vec سؤالين مفتوحين.

أولاً، كان هناك خط موازٍ من البحث قام بتحليل مصفوفة التواجد المشترك مباشرةً (LSA، HAL) بدلاً من إجراء تحديثات تخطي غرام عبر الإنترنت. هل كان النهج التكراري لـ Word2Vec أفضل بشكل أساسي، أم أن الاختلاف كان نتيجة لكيفية تعامل الطريقتين مع الأعداد؟ أجاب **GloVe** على ذلك: تحليل المصفوفة مع خسارة مختارة بعناية تتطابق مع Word2Vec أو تتفوق عليه، كما أن تكاليف التدريب أقل.

ثانيًا، لم يكن لدى أي من الطريقتين قصة للكلمات التي لم يسبق لها مثيل. `Zoomer-approved`، `dogecoin`، أي اسم علم تمت صياغته الأسبوع الماضي، كل شكل تصريف لجذر نادر. **FastText** أصلح هذا عن طريق تضمين حرف n-gram: الكلمة هي مجموع أجزائها، بما في ذلك المقاطع، لذلك حتى الكلمات التي لا تحتوي على مفردات تحصل على ناقل معقول.

ثالثًا، بمجرد وصول المحولات، تغير السؤال مرة أخرى. تصل المفردات على مستوى الكلمة إلى حوالي مليون إدخال؛ اللغة الحقيقية أكثر انفتاحًا من ذلك. **ترميز زوج البايت (BPE)** وحل هذا الأمر بأقاربه من خلال تعلم مفردات من وحدات الكلمات الفرعية المتكررة التي تغطي كل شيء. كل رمز مميز حديث لكل LLM حديث هو رمز مميز للكلمات الفرعية.

يتناول هذا الدرس العناصر الثلاثة، ثم يشرح ما يجب الوصول إليه ومتى.

## The Concept

**GloVe (المتجهات العالمية).** أنشئ مصفوفة التكرار المشترك للكلمة والكلمة `X` حيث `X[i][j]` هو عدد مرات ظهور الكلمة `j` في سياق الكلمة `i`. تدريب المتجهات مثل `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. وزن الخسارة بحيث لا تهيمن الأزواج المتكررة. منتهي.

**FastText.** الكلمة هي مجموع أحرفها n-grams بالإضافة إلى الكلمة نفسها. `where` يصبح `<wh, whe, her, ere, re>, <where>`. متجه الكلمة هو مجموع تلك المتجهات المكونة. تدريب كـ Word2Vec. فائدة: كلمات غيبية (`whereupon`) مؤلفة من ن جرامات معروفة.

**BPE (ترميز زوج البايت).** ابدأ بمفردات البايتات الفردية (أو الأحرف). عد كل زوج مجاور في المجموعة. قم بدمج الزوج الأكثر تكرارًا في رمز مميز جديد. كرر ذلك لـ `k` التكرارات. النتيجة: مفردات من الرموز `k + 256` حيث تكون التسلسلات المتكررة (`ing`، `tion`، `the`) عبارة عن رموز فردية ويتم تقسيم الكلمات النادرة إلى أجزاء مألوفة. كل جملة ترمز إلى شيء ما.

## Build It

### GloVe: factorize the co-occurrence matrix

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

قطعتان متحركتان تستحقان التسمية. تقوم وظيفة الترجيح `f(x) = (x/x_max)^alpha` بتخفيض الوزن للأزواج المتكررة جدًا (مثل `(the, and)`) بحيث لا تهيمن على الخسارة. التضمين النهائي هو مجموع جدولي `W` (الوسط) و`W_tilde` (السياق). إن جمع كليهما عبارة عن خدعة منشورة تميل إلى التفوق في الأداء باستخدام واحدة فقط.

### FastText: subword-aware embeddings

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

يتم تمثيل كل كلمة بمجموعتها من n-grams (عادة من 3 إلى 6 أحرف). تضمين الكلمة هو مجموع تضمينات n-gram الخاصة بها. للتدريب على تخطي جرام، قم بتوصيل هذا حيث يستخدم Word2Vec متجهًا واحدًا.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

بالنسبة للكلمة غير المرئية، لا يزال بإمكانك الحصول على متجه طالما أن بعض جراماتها n معروفة. `whereupon` سهم `<wh`، `her`، `ere`، و `<where` مع `where`، بحيث يهبط الاثنان بالقرب من بعضهما البعض.

### BPE: learned subword vocabulary

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

يدمج التكرار الأول الزوج المجاور الأكثر شيوعًا. بعد تكرارات كافية، تصبح السلاسل الفرعية المتكررة (`low`، `est`، `tion`) رموزًا فردية وتنكسر الكلمات النادرة بشكل نظيف.

تتعلم الرموز المميزة GPT / BERT / T5 عمليات الدمج من 30 ألف إلى 100 ألف. النتيجة: تحويل أي نص إلى تسلسل محدد الطول من المعرفات المعروفة، لا يوجد OOV على الإطلاق.

## Use It

في الممارسة العملية، نادرًا ما تقوم بتدريب أي من هؤلاء بنفسك. يمكنك تحميل نقاط التفتيش المدربة مسبقا.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

لترميز الكلمات الفرعية على النمط BPE في عصر المحولات:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

تشير البادئة `Ġ` إلى حدود الكلمات (اتفاقية GPT-2). كل رمز مميز حديث هو BPE متغير، WordPiece (BERT)، أو SentencePiece (T5، LLaMA).

### When to pick which

| الوضع | اختر |
|-----------|------|
| ناقلات الكلمات ذات الأغراض العامة المدربة مسبقًا، لا حاجة إلى تسامح OOV | قفاز 300د |
| نواقل الكلمات ذات الأغراض العامة المدربة مسبقًا، يجب أن تتعامل مع الأخطاء الإملائية / الألفاظ الجديدة / اللغات الغنية شكليًا | نص سريع |
| أي شيء يدخل في محول (التدريب أو الاستدلال) | مهما كان الرمز المميز الذي يتم شحن النموذج معه. أبدا مبادلة. |
| تدريب نموذج اللغة الخاص بك من الصفر | قم بتدريب رمز BPE أو SentencePiece على مجموعتك أولاً |
| تصنيف نص الإنتاج بالنموذج الخطي | لا يزال TF-IDF. الدرس 02. |

## Ship It

حفظ باسم `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Exercises

1. **سهل.** قم بتشغيل `char_ngrams("playing")` و`char_ngrams("played")`. احسب تداخل Jaccard لمجموعتي n-gram. يجب أن تشاهد أجزاء كبيرة مشتركة (`pla`، `lay`، `play`)، ولهذا السبب ينتقل FastText بشكل جيد عبر المتغيرات المورفولوجية.
2. **متوسط.** قم بتمديد `learn_bpe` لتتبع نمو المفردات. ارسم الرموز المميزة لكل مجموعة أحرف كدالة لعدد عمليات الدمج. من المفترض أن تشاهد ضغطًا سريعًا في البداية، حيث يقترب من 2-3 أحرف تقريبًا لكل رمز مميز.
3. **صعب.** تدريب دمج 1k BPE على أعمال شكسبير الكاملة. قارن بين ترميز الكلمات الشائعة وأسماء العلم النادرة. قياس متوسط ​​الرموز لكل كلمة قبل وبعد. اكتب ما فاجأك.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| مصفوفة التواجد المشترك | جدول تكرار الكلمات والكلمات | `X[i][j]` = عدد مرات ظهور الكلمة `j` في النافذة حول الكلمة `i`. |
| الكلمة الفرعية | قطعة من كلمة | حرف n-gram (FastText) أو الرمز المميز (BPE/WordPiece/SentencePiece). |
| BPE | ترميز زوج البايت | الدمج التكراري للأزواج المتجاورة الأكثر تكرارًا حتى تصل المفردات إلى الحجم المستهدف. |
| OOV | خارج المفردات | كلمة النموذج لم يسبق له مثيل. فشل Word2Vec/GloVe. FastText و BPE التعامل معها. |
| مستوى البايت BPE | BPE على البايتات الخام | مخطط GPT-2. تبدأ المفردات بـ 256 بايت، لذا لا يوجد شيء على الإطلاق OOV. |

## Further Reading

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — the GloVe paper, seven pages, still the best derivation of the loss.
- [Bojanowski et al. (2017). إثراء متجهات الكلمات بمعلومات الكلمات الفرعية](https://arxiv.org/abs/1607.04606) — FastText.
- [سينريش، هادو، بيرش (2016). الترجمة الآلية العصبية للكلمات النادرة مع وحدات الكلمات الفرعية](https://arxiv.org/abs/1508.07909) — الورقة التي قدمت BPE إلى NLP الحديثة.
- [Hugging Face ملخص الرمز المميز](https://huggingface.co/docs/transformers/tokenizer_summary) — كيف تختلف BPE وWordPiece وSentencePiece فعليًا في الممارسة العملية.
