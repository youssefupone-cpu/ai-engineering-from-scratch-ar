#تحليل المشاعر
> المهمة NLP الأساسية. يظهر هنا معظم ما تحتاج إلى معرفته حول تصنيف النص الكلاسيكي.
**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** المرحلة 5 · 02 (BoW + TF-IDF)، المرحلة 2 · 14 (Naive Bayes)
**الوقت:** ~75 دقيقة
## المشكلة
"الطعام لم يكن رائعًا." إيجابية أم سلبية؟
المشاعر تبدو بسيطة. قال أحد المراجعين إنهم أحبوا أو لم يعجبهم شيء ما. قم بتسمية الجملة. السبب الذي جعلها تصبح مهمة NLP الأساسية هو أن كل حالة سهلة المظهر تخفي حالة صعبة. النفي يقلب المعنى السخرية تقلبها. "ليس سيئًا على الإطلاق" هو ​​أمر إيجابي على الرغم من وجود كلمتين مشفرة سلبيًا. تحمل الرموز التعبيرية إشارة أكثر من النص المحيط بها. مفردات المجال مهمة (`tight` في مراجعة الموسيقى مقابل `tight` في مراجعة الموضة).
المشاعر هي مختبر عمل لـ NLP الكلاسيكي. إذا فهمت لماذا يحتوي كل خط أساس ساذج على وضع فشل محدد، فإنك تفهم سبب اختراع كل نموذج أكثر ثراءً. يبني هذا الدرس خط أساس Naive Bayes من الصفر، ويضيف الانحدار اللوجستي، ويسمي المصائد التي تجعل مشاعر الإنتاج make مشكلة في درجة الامتثال.
##المفهوم
المشاعر الكلاسيكية هي وصفة من خطوتين.
1. **تمثيل.** تحويل النص إلى ناقل الميزة. BoW، TF-IDF، أو n-grams.
2. **التصنيف.** قم بمطابقة النموذج الخطي (Naive Bayes، الانحدار اللوجستي، SVM) على الأمثلة المصنفة.
Naive Bayes هو أغبى نموذج ناجح. افترض أن كل ميزة مستقلة بالنظر إلى التسمية. قم بتقدير `P(word | positive)` و`P(word | negative)` من الأعداد. عند الاستدلال، اضرب الاحتمالات. إن افتراض الاستقلال "الساذج" خاطئ إلى حد مثير للضحك، ومع ذلك فإن نتائجه قوية إلى حد صادم. السبب: مع ميزات النص المتفرقة والبيانات المعتدلة، يهتم المصنف بالجانب الذي تميل إليه كل كلمة أكثر من اهتمامه بالجانب الذي تميل إليه كل كلمة.
يعمل الانحدار اللوجستي على إصلاح افتراض الاستقلال. ويتعرف على الوزن لكل ميزة، بما في ذلك الأوزان السلبية. `not good` كميزة بيجرام تحصل على وزن سلبي. لا يستطيع Naive Bayes فعل ذلك بالنسبة للبيغرامات التي لم يتم تصنيفها من قبل.
## بنائها
### الخطوة 1: مجموعة بيانات مصغرة حقيقية
```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

صغيرة عن قصد. يستخدم العمل الحقيقي عشرات الآلاف من الأمثلة (IMDb، SST-2، قطبية Yelp). الرياضيات متطابقة.
### الخطوة 2: متعدد الحدود Naive Bayes من الصفر
```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

التجانس الإضافي (alpha=1.0) هو تجانس لابلاس. وبدون ذلك، فإن الكلمة غير المرئية في الفصل الدراسي يكون احتمالها صفرًا وينفجر السجل. `alpha=0.01` شائع في الممارسة العملية. `alpha=1.0` هو الإعداد الافتراضي للتدريس.
### الخطوة 3: الانحدار اللوجستي من الصفر
```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 التنظيم مهم هنا. ميزات النص متفرقة؛ بدون L2 يحفظ النموذج أمثلة التدريب. ابدأ عند `0.01` وقم بالضبط.
### الخطوة 4: التعامل مع النفي (وضع الفشل)
فكر في "ليس جيدًا" و"ليس سيئًا". يرى مصنف BoW `{not, good}` و`{not, bad}` ويتعلم من أيهما ظهر أكثر في التدريب. يرى مصنف بيجرام `not_good` و`not_bad` ويتعرف عليهما كميزات مميزة. وهذا عادة ما يكون كافيا.
إصلاح أكثر بدائية يعمل عندما لا يكون لديك صور كبيرة: **نطاق النفي**. رموز البادئة التي تتبع كلمة النفي مع `NOT_` حتى علامة الترقيم التالية.
```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

الآن أصبح `good` و`NOT_good` ميزات مختلفة. يمكن للمصنف أن يزنهم عكس ذلك. ثلاثة خطوط من المعالجة المسبقة والدقة القابلة للقياس تقفز على معايير المشاعر.
### الخطوة 5: مقاييس التقييم المهمة
الدقة وحدها تكون مضللة إذا كانت الطبقات غير متوازنة. عادة ما تكون المشاعر الحقيقية إيجابية بنسبة 70-80% أو سلبية بنسبة 70-80%؛ يحصل المصنف ذو الأغلبية الثابتة على دقة تبلغ 80٪ ولا قيمة له. الإبلاغ عن كل مما يلي:
- **الدقة والتذكر لكل فصل.** زوج واحد لكل فصل. قم بحساب المتوسط ​​الكلي لها للحصول على رقم واحد يحترم توازن الفصل.
- **ماكرو-F1 (المقياس الأساسي للبيانات غير المتوازنة).** متوسط ​​درجات F1 لكل فئة، مرجحة بالتساوي. استخدم هذا بدلاً من الدقة عندما تكون الفئات غير متوازنة.
- **المرجح-F1 (بديل).** نفس الماكرو ولكن مرجح حسب تكرار الفئة. قم بالإبلاغ جنبًا إلى جنب مع الماكرو-F1 عندما يكون لاختلال التوازن نفسه معنى تجاري.
- **مصفوفة الارتباك.** الأعداد الأولية. افحص دائمًا قبل الثقة في أي مقياس عددي؛ فهو يكشف عن أي زوج من الفئات يخلط النموذج.
- **عينات الأخطاء لكل فصل.** سحب 5 توقعات خاطئة لكل فصل. اقرأها. لا شيء يحل محل قراءة الأخطاء الفعلية.
بالنسبة للبيانات غير المتوازنة بشدة (> نسبة 95-5)، قم بالإبلاغ عن **AUROC** و **AUPRC** بدلاً من الدقة. AUPRC أكثر حساسية تجاه فئة الأقلية، وهو ما تهتم به عادةً (البريد العشوائي، والاحتيال، والمشاعر النادرة).
**خطأ شائع يجب تجنبه.** إن الإبلاغ عن micro-F1 بدلاً من Macro-F1 بشأن البيانات غير المتوازنة يعطي رقمًا يبدو مرتفعًا لأنه تهيمن عليه فئة الأغلبية. Macro-F1 يجبرك على رؤية أداء فئة الأقلية.
```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## استخدمه
scikit-learn يفعل ذلك في ستة أسطر بشكل صحيح.
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

ثلاثة أشياء يجب ملاحظتها. `stop_words=None` يحتفظ بالنفي. يضيف `ngram_range=(1, 2)` صورًا كبيرة بحيث يصبح `not_good` ميزة. `sublinear_tf=True` يخفف الكلمات المتكررة. تمثل هذه العلامات الثلاثة الفرق بين خط الأساس الدقيق بنسبة 75% وخط الأساس الدقيق بنسبة 85% في SST-2.
### متى يجب الوصول إلى المحول
- كشف السخرية. النماذج الكلاسيكية تفشل هنا. فترة.
- مراجعات طويلة حيث تتغير المشاعر في منتصف المستند.
- المشاعر القائمة على الجانب. "الكاميرا كانت رائعة ولكن البطارية كانت فظيعة." تحتاج إلى أن تنسب المشاعر إلى الجوانب. المحولات أو نماذج الإخراج المنظمة فقط.
- لغات غير الإنجليزية ومنخفضة الموارد. متعدد اللغات BERT يمنحك خط الأساس الصفري مجانًا.
إذا كنت بحاجة إلى أي مما سبق، فانتقل إلى المرحلة السابعة (الغوص العميق في المحولات). بخلاف ذلك، فإن Naive Bayes أو الانحدار اللوجستي في TF-IDF بالإضافة إلى bigrams بالإضافة إلى التعامل مع النفي هو خط الأساس للإنتاج لعام 2026.
### فخ التكاثر (مرة أخرى)
إن إعادة تدريب نماذج المشاعر أمر روتيني. إعادة تقييم لهم ليست كذلك. تستخدم أرقام الدقة المذكورة في الأوراق تقسيمات محددة، ومعالجة مسبقة محددة، ورموز مميزة محددة. إذا قمت بمقارنة نموذجك الجديد بخط الأساس دون استخدام خط pipe المطابق، فسوف تحصل على دلتا مضللة. قم دائمًا بإعادة إنشاء خط الأساس على pipeline، وليس رقم الورقة.
## اشحنها
حفظ باسم `outputs/prompt-sentiment-baseline.md`:
```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## تمارين
1. **سهل.** أضف `apply_negation` كخطوة معالجة مسبقة في سطر scikit-learn pipe وقم بقياس دلتا F1 في مجموعة بيانات مشاعر صغيرة.
2. **متوسط.** قم بتنفيذ الانحدار اللوجستي المرجح حسب الفئة (مرر `class_weight="balanced"` إلى scikit-learn، أو استنتج التدرج اللوني بنفسك). قياس التأثير على اختلال التوازن الطبقي 90-10.
3. **صعب.** قم ببناء كاشف للسخرية من خلال تدريب مصنف ثانٍ على بقايا نموذج المشاعر. توثيق الإعداد التجريبي الخاص بك. حذر القارئ عندما تكون دقتك أقل من الصدفة (مستوى الصدفة في السخرية من الدرجة الثانية يصل إلى 50% تقريبًا، ومعظم المحاولات الأولى تصل إلى هناك).
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|-|--------------------------------------|
| قطبية | إيجابي أم سلبي | تسمية ثنائية؛ يمتد أحيانًا إلى محايد أو دقيق الحبيبات (5 نجوم). |
| المشاعر القائمة على الجانب | قطبية لكل جانب | إسناد المشاعر إلى كيانات أو سمات محددة مذكورة في النص. |
| نطاق النفي | عكس الرموز القريبة | رموز البادئة بعد "not" مع `NOT_` حتى علامة الترقيم. |
| تجانس لابلاس | إضافة 1 إلى الأعداد | يمنع ميزات الاحتمال الصفري في Naive Bayes. |
| L2 التسوية | تقليص الأوزان | يضيف `lambda * sum(w^2)` إلى الخسارة. ضروري لميزات النص المتناثر. |
## مزيد من القراءة
- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — المسح التأسيسي. طويلة، ولكن الأقسام الأربعة الأولى تغطي كل شيء كلاسيكي.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — الورقة التي أظهرت الصور الكبيرة + Naive Bayes من الصعب التغلب عليها في النص القصير.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — مرجع لـ `CountVectorizer`، `TfidfVectorizer`، وكل مقبض ستقوم بضبطه.