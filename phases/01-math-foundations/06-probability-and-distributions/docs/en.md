# الاحتمال والتوزيعات

> الاحتمالية هي اللغة التي يستخدمها الذكاء الاصطناعي للتعبير عن عدم اليقين.

**النوع:** تعلم
** اللغة: ** بايثون
**المتطلبات الأساسية:** المرحلة الأولى، الدروس 01-04
**الوقت:** ~75 دقيقة

## أهداف التعلم

- تنفيذ PMFs وPDFs من الصفر لتوزيعات برنولي والفئوية وبواسون والموحدة والعادية
- حساب القيمة المتوقعة والتباين واستخدام نظرية الحد المركزي لشرح سبب هيمنة الغاوسيين
- إنشاء وظائف softmax وlog-softmax باستخدام خدعة الاستقرار الرقمي (طرح max logit)
- حساب الخسارة عبر الإنتروبيا من اللوغاريتمات وربطها باحتمال السجل السلبي

## المشكلة

يقوم المصنف بإخراج `[0.03, 0.91, 0.06]`. يختار نموذج اللغة الكلمة التالية من بين 50000 مرشح. يقوم نموذج الانتشار بإنشاء صور عن طريق أخذ عينات من التوزيعات المستفادة. كل هذه احتمالات في العمل.

كل تنبؤ يقوم به النموذج هو توزيع احتمالي. تقيس كل دالة خسارة مدى بعد التوزيع المتوقع عن التوزيع الحقيقي. تقوم كل خطوة تدريب بضبط المعلمات لجعل التوزيع يبدو أشبه بتوزيع آخر. بدون احتمال، لا يمكنك قراءة ورقة واحدة لتعلم الآلة، أو تصحيح نموذج واحد، أو فهم سبب كون خسارة التدريب الخاصة بك هي NaN.

##المفهوم

### الأحداث، ومساحات العينة، والاحتمالات

فضاء العينة S هو مجموعة النواتج الممكنة الحدث هو مجموعة فرعية من فضاء العينة. تقوم الاحتمالات بتعيين الأحداث إلى أرقام بين 0 و 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

تحدد ثلاث بديهيات كل الاحتمالات:
1. P(A) >= 0 لأي حدث A
2. P(S) = 1 (يحدث شيء ما دائمًا)
3. P(A أو B) = P(A) + P(B) عندما لا يمكن أن يحدث A وB معًا

كل شيء آخر (نظرية بايز، التوقعات، التوزيعات) يتبع من هذه القواعد الثلاث.

### الاحتمالية المشروطة والاستقلال

P(A|B) هو احتمال A بشرط حدوث B.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

هناك حدثان مستقلان عندما تعرف أن أحدهما لا يخبرك شيئًا عن الآخر:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

تقلب العملة مستقلة. بطاقات الرسم دون استبدال ليست كذلك.

### دوال الكتلة الاحتمالية مقابل دوال الكثافة الاحتمالية

المتغيرات العشوائية المنفصلة لها دالة احتمالية (PMF). كل نتيجة لها احتمالية محددة يمكنك قراءتها مباشرة.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

المتغيرات العشوائية المستمرة لها دالة كثافة الاحتمال (PDF). الكثافة عند نقطة واحدة ليست احتمالا. تأتي الاحتمالية من تكامل الكثافة خلال فترة زمنية.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

هذا التمييز مهم في ML. مخرجات التصنيف هي PMFs (اختيارات منفصلة). تستخدم المساحات الكامنة VAE ملفات PDF (مستمرة).

### التوزيعات المشتركة

** برنولي: ** تجربة واحدة، نتيجتان. نماذج التصنيف الثنائي.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**فئوي:** تجربة واحدة، نتائج k. نماذج تصنيف متعدد الفئات (إخراج softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**الزي الرسمي:** جميع النتائج محتملة على قدم المساواة. تستخدم للتهيئة العشوائية.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

** عادي (غاوسي): ** منحنى الجرس. يتم تحديد المعلمات بواسطة المتوسط ​​(mu) والتباين (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**بواسون:** عدد الأحداث النادرة في فترة زمنية محددة. معدلات أحداث النماذج.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### القيمة المتوقعة والتباين

القيمة المتوقعة هي نتيجة المتوسط ​​المرجح.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

تنتشر مقاييس التباين حول المتوسط.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

في تعلم الآلة، تظهر القيمة المتوقعة كدالة الخسارة (متوسط ​​الخسارة عبر توزيع البيانات). يخبرك التباين عن استقرار النموذج. التباين العالي في التدرجات يعني التدريب الصاخب.

### التوزيعات المشتركة والهامشية

يصف التوزيع المشترك P(X, Y) متغيرين عشوائيين معًا.

مثال PMF مشترك (X = الطقس، Y = المظلة):

| | Y=0 (بدون مظلة) | Y=1 (مظلة) | هامشي P(X) |
|---|---|---|---|
| X=0 (الشمس) | 0.40 | 0.10 | ف(X=0) = 0.50 |
| X=1 (مطر) | 0.05 | 0.45 | ف(X=1) = 0.50 |
| **الهامشي P(Y)** | ف(ص=0) = 0.45 | ف(ص=1) = 0.55 | 1.00 |

التوزيع الهامشي يلخص المتغير الآخر:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

إجماليات الصفوف والأعمدة في الجدول أعلاه هي الهوامش.

### لماذا يظهر التوزيع الطبيعي في كل مكان

نظرية الحد المركزي: مجموع (أو متوسط) العديد من المتغيرات العشوائية المستقلة يتقارب إلى التوزيع الطبيعي، بغض النظر عن التوزيع الأصلي.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

لهذا السبب:
- أخطاء القياس طبيعية تقريبًا (العديد من المصادر المستقلة الصغيرة)
- تهيئة الوزن في الشبكات العصبية تستخدم التوزيعات العادية
- الضوضاء المتدرجة في SGD طبيعية تقريبًا (مجموع العديد من تدرجات العينة)
- التوزيع الطبيعي هو الحد الأقصى لتوزيع الإنتروبيا لمتوسط وتباين معين

### احتمالات السجل

الاحتمالات الأولية تسبب مشاكل عددية. إن ضرب العديد من الاحتمالات الصغيرة معًا يتدفق بسرعة إلى الصفر.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

احتمالات السجل تصلح هذا. الضرب يصبح إضافات.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

القواعد:
- سجل (أ * ب) = سجل (أ) + سجل (ب)
- احتمالات السجل دائمًا <= 0 (بما أن 0 < P <= 1)
- أكثر سلبية = أقل احتمالا
- الخسارة عبر الإنتروبيا هي احتمال السجل السلبي للفئة الصحيحة

### Softmax كتوزيع احتمالي

تقوم الشبكات العصبية بإخراج النتائج الأولية (السجلات). يقوم Softmax بتحويلها إلى توزيع احتمالي صالح.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

خدعة softmax: اطرح الحد الأقصى من السجل قبل الأس لمنع التجاوز.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

يجمع Log-softmax بين softmax وlog لتحقيق الاستقرار العددي. يستخدم PyTorch هذا داخليًا لخسارة الإنتروبيا المتقاطعة.

### أخذ العينات

أخذ العينات يعني رسم قيم عشوائية من التوزيع. في مل:
- التسرب عينات عشوائية من الخلايا العصبية إلى الصفر
- زيادة بيانات عينات التحويلات العشوائية
- نماذج اللغة تختبر الرمز المميز التالي من التوزيع المتوقع
- تقوم نماذج الانتشار بأخذ عينات من الضوضاء وتقليل الضوضاء تدريجيًا

يتطلب أخذ العينات من التوزيعات التعسفية تقنيات مثل أخذ عينات التحويل العكسي، أو أخذ عينات الرفض، أو خدعة إعادة المعلمة (المستخدمة في VAEs).

## بنائها

### الخطوة 1: أساسيات الاحتمالية

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### الخطوة 2: PMF وPDF من البداية

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### الخطوة الثالثة: القيمة المتوقعة والتباين

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### الخطوة 4: أخذ العينات من التوزيعات

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### الخطوة 5: احتمالات Softmax والسجل

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### الخطوة 6: عرض نظرية الحد المركزي

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### الخطوة 7: التصور

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

التنفيذ الكامل مع كافة المرئيات موجود في `code/probability.py`.

## استخدمه

مع NumPy وSciPy، كل ما سبق هو مجرد سطر واحد:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

لقد بنيت هذه من الصفر. الآن أنت تعرف ما تفعله مكالمات المكتبة.

## تمارين

1. تنفيذ أخذ عينات التحويل العكسي للتوزيع الأسي. تحقق من ذلك عن طريق أخذ عينات من 10000 قيمة ومقارنة الرسم البياني بملف PDF الحقيقي.

2. بناء جدول توزيع مشترك لنردين محملين. احسب التوزيعات الهامشية وتحقق مما إذا كان النرد مستقلاً.

3. احسب الخسارة عبر الإنتروبيا لمصنف من 5 فئات يقوم بإخراج السجلات `[2.0, 0.5, -1.0, 3.0, 0.1]` عندما تكون الفئة الصحيحة هي الفهرس 3. ثم تحقق من إجابتك باستخدام `nn.CrossEntropyLoss` الخاص بـ PyTorch.

4. اكتب دالة تأخذ قائمة باحتمالات السجل وترجع التسلسل الأكثر احتمالاً، وإجمالي احتمال السجل، والاحتمال الأولي المكافئ. اختبرها بجملة من 50 كلمة حيث احتمال كل كلمة 0.01.

## المصطلحات الرئيسية

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| مساحة العينة | "كل الاحتمالات" | المجموعة S لكل نتيجة محتملة للتجربة |
| قوات الدفاع الشعبي | "دالة الاحتمال" | دالة تعطي الاحتمال الدقيق لكل نتيجة منفصلة، ​​ومجموعها 1 |
| قوات الدفاع الشعبي | "منحنى الاحتمالية" | دالة الكثافة للمتغيرات المستمرة. دمجها على فترة زمنية للحصول على الاحتمال |
| الاحتمال الشرطي | "الاحتمال المعطى لشيء ما" | P(A\|B) = P(A وB) / P(B). أسس التفكير البايزي ونظرية بايز |
| الاستقلال | "لا يؤثر بعضهم في بعض" | ف(أ وب) = ف(أ) * ف(ب). معرفة حدث واحد لا تخبرك شيئًا عن الآخر |
| القيمة المتوقعة | "المتوسط" | المجموع المرجح لجميع النتائج. دالة الخسارة هي قيمة متوقعة |
| التباين | "كيف انتشرت" | الانحراف التربيعي المتوقع عن المتوسط. التباين العالي = تقديرات صاخبة وغير مستقرة |
| التوزيع الطبيعي | "منحنى الجرس" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). يظهر في كل مكان بسبب CLT |
| نظرية الحد المركزي | "المتوسطات تصبح طبيعية" | يتقارب متوسط ​​العديد من العينات المستقلة مع التوزيع الطبيعي بغض النظر عن المصدر |
| التوزيع المشترك | "متغيران معًا" | تصف P(X, Y) احتمالية كل مجموعة من نتائج X وY |
| التوزيع الهامشي | "اجمع المتغير الآخر" | P(X) = sum_y P(X, Y). يستعيد توزيع متغير واحد من المفصل |
| احتمالية السجل | "سجل الاحتمال" | سجل ف (س). يحول المنتجات إلى مبالغ، ويمنع التدفق العددي في تسلسلات طويلة |
| سوفت ماكس | "تحويل النتائج إلى احتمالات" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). يعين اللوغاريتمات ذات القيمة الحقيقية لتوزيع احتمالي صالح |
| عبر الانتروبيا | "وظيفة الخسارة" | -sum(p_true * log(p_predicted)). يقيس مدى اختلاف التوزيعتين. الدنيا أفضل |
| اللوجيستات | "مخرجات النموذج الخام" | درجات غير طبيعية قبل softmax. سميت على اسم الوظيفة اللوجستية |
| أخذ العينات | "رسم قيم عشوائية" | توليد القيم وفقا للتوزيع الاحتمالي. كيف تولد النماذج المخرجات |

## مزيد من القراءة

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - دليل مرئي على سبب تحول المتوسطات إلى مستويات طبيعية
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - مرجع موجز يغطي كل شيء هنا وأكثر
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - ما أهمية الاستقرار العددي وكيفية تحقيقه