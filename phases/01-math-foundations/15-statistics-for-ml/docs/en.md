# Statistics for Machine Learning

> الإحصائيات هي الطريقة التي تعرف بها ما إذا كان نموذجك يعمل بالفعل أم أنه محظوظ.

**النوع:** بناء
** اللغة: ** بايثون
**المتطلبات:** المرحلة الأولى، الدروس 06 (الاحتمالات والتوزيعات)، 07 (نظرية بايز)
**الوقت:** ~120 دقيقة

## Learning Objectives

- حساب الإحصاء الوصفي، وارتباط بيرسون/سبيرمان، ومصفوفات التغاير من الصفر
- إجراء اختبارات الفرضيات (اختبار t، مربع كاي) وتفسير القيم الاحتمالية وفترات الثقة بشكل صحيح
- استخدم إعادة تشكيل bootstrap لإنشاء فترات ثقة لأي مقياس دون افتراضات توزيعية
- تمييز الدلالة الإحصائية عن الدلالة العملية باستخدام مقاييس حجم التأثير

## The Problem

لقد قمت بتدريب نموذجين. حصل النموذج A على 0.87 في مجموعة الاختبار الخاصة بك. النموذج ب يسجل 0.89. قمت بنشر النموذج ب. وبعد ثلاثة أسابيع، أصبحت مقاييس الإنتاج أسوأ من ذي قبل. ماذا حدث؟

لم يتفوق النموذج B فعليًا على النموذج A. وكان الفارق 0.02 هو الضوضاء. كانت مجموعة الاختبار الخاصة بك صغيرة جدًا، أو كان التباين مرتفعًا جدًا، أو كليهما. لقد شحنت العشوائية بملابس التحسين.

يحدث هذا باستمرار. تغييرات لوحة المتصدرين في Kaggle. الأوراق التي تفشل في التكاثر. اختبارات A/B التي تعلن الفائزين بناءً على بضع مئات من العينات. السبب الجذري هو نفسه دائمًا: لقد تخطى شخص ما الإحصائيات.

تمنحك الإحصائيات الأدوات اللازمة لتمييز الإشارة عن الضوضاء. فهو يخبرك عندما يكون الاختلاف حقيقيًا، ومدى الثقة التي يجب أن تكون عليها، وكم البيانات التي تحتاجها قبل أن تتمكن من الوثوق بالنتيجة. كل ML pipeline، كل مقارنة للنماذج، كل تجربة تحتاج إلى إحصائيات. بدونها، أنت تخمن.

## The Concept

### Descriptive Statistics: Summarizing Your Data

قبل أن تصمم أي شيء، عليك أن تعرف كيف تبدو بياناتك. تعمل الإحصائيات الوصفية على ضغط مجموعة البيانات في عدد قليل من الأرقام التي تلتقط شكلها.

**مقاييس النزعة المركزية** تجيب على "أين يقع الوسط؟"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

المتوسط ​​هو نقطة التوازن. الوسيط هو علامة المنتصف. عندما تتباعد، يكون توزيعك منحرفًا. توزيعات الدخل لها متوسط ​​>> متوسط ​​(انحراف صحيح عن المليارديرات). غالبًا ما يكون لتوزيعات الخسارة أثناء التدريب متوسط ​​<< متوسط ​​(الانحراف الأيسر من العينات السهلة).

**مقاييس الانتشار** تجيب على "ما مدى تشتت البيانات؟"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**النسب المئوية** تقسم البيانات المصنفة إلى 100 جزء متساوي. المئين الخامس والعشرون (Q1) يعني أن 25% من القيم تقع تحت هذه النقطة. المئين الخمسين هو المتوسط. النسبة المئوية الخامسة والسبعون هي Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

في ML، تهتم بالنسب المئوية لزمن وصول الاستدلال، وتوزيعات ثقة التنبؤ، وفهم توزيعات الخطأ. قد يكون النموذج ذو متوسط ​​الخطأ المنخفض ولكن الخطأ الفادح P99 عديم الفائدة للتطبيقات ذات الأهمية الحيوية للسلامة.

**إحصائيات العينة مقابل إحصائيات السكان.** عند حساب التباين من عينة، قم بالقسمة على (n-1) بدلاً من n. هذا هو تصحيح بيسل. إنه يعوض عن حقيقة أن متوسط ​​العينة الخاص بك ليس هو متوسط ​​السكان الحقيقي. مع وجود n في المقام، فإنك تقلل بشكل منهجي من التباين الحقيقي. مع (n-1)، يكون التقدير غير متحيز.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

من الناحية العملية: إذا كانت n كبيرة (آلاف العينات)، يكون الفرق ضئيلًا. إذا كانت n صغيرة (عشرات العينات)، فهذا مهم.

### Correlation: How Variables Move Together

يقيس الارتباط قوة واتجاه العلاقة الخطية بين متغيرين.

**معامل ارتباط بيرسون** يقيس الارتباط الخطي:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

يفترض بيرسون أن العلاقة خطية وأن كلا المتغيرين يتم توزيعهما بشكل طبيعي تقريبًا. إنها حساسة للقيم المتطرفة. يمكن لنقطة متطرفة واحدة أن تسحب r من 0.1 إلى 0.9.

**ارتباط رتبة سبيرمان** يقيس الارتباط الرتيب:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**متى يستخدم كل منهما:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**القاعدة الذهبية:** الارتباط لا يعني السببية. ترتبط مبيعات الآيس كريم ووفيات الغرق لأن كلاهما يزداد في الصيف. هناك ارتباط بين دقة النموذج وعدد المعلمات، لكن إضافة المعلمات لا يؤدي إلى تحسين الدقة تلقائيًا (راجع: التجهيز الزائد).

### Covariance Matrix

يقيس التباين بين متغيرين مدى اختلافهما معًا:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

بالنسبة للميزات d، تكون مصفوفة التغاير C هي مصفوفة d x d حيث C[i][j] = Cov(feature_i, feature_j). الإدخالات القطرية C[i][i] هي الفروق في كل ميزة.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

** الاتصال بـ PCA.** PCA eigen يتحلل مصفوفة التغاير. المتجهات الذاتية هي المكونات الرئيسية (اتجاهات التباين الأقصى). تخبرك القيم الذاتية بمقدار التباين الذي يلتقطه كل مكون. هذا هو بالضبط ما تناوله الدرس 10، ولكنك الآن ترى لماذا تعتبر مصفوفة التغاير هي الشيء الصحيح الذي يجب تحليله: فهي تقوم بتشفير جميع العلاقات الخطية الزوجية في بياناتك.

**الاتصال بالارتباط.** مصفوفة الارتباط هي مصفوفة التغاير للمتغيرات القياسية (كل منها مقسوم على انحرافه المعياري). يعمل الارتباط على تطبيع التباين بحيث تقع جميع القيم في [-1، 1].

### Hypothesis Testing

اختبار الفرضية هو إطار لاتخاذ القرارات في ظل عدم اليقين. تبدأ بمطالبة، وتجمع البيانات، وتحدد ما إذا كانت البيانات متوافقة مع المطالبة.

**الإعداد:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**القيمة الاحتمالية** هي احتمالية رؤية بيانات متطرفة مثل ما لاحظته، بافتراض أن H0 صحيحة. إنه NOT احتمال أن H0 صحيح. هذا هو سوء الفهم الأكثر شيوعًا في الإحصائيات.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**فواصل الثقة** تعطي نطاقًا من القيم المعقولة للمعلمة:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

يخبرك عرض فاصل الثقة بالدقة. الفواصل الزمنية الواسعة تعني عدم اليقين العالي. تعني الفواصل الزمنية الضيقة أن تقديرك دقيق (ولكن ليس بالضرورة دقيقًا، إذا كانت بياناتك متحيزة).

### The t-test

اختبار t يقارن الوسائل. هناك عدة نكهات.

**اختبار t لعينة واحدة:** هل يختلف متوسط ​​المجتمع عن القيمة المفترضة؟

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**اختبار (ت) لعينتين (مستقل):** هل مجموعتان تعنيان مختلفتين؟

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**اختبار t المقترن:** عندما تأتي القياسات في أزواج (يتم تقييم نفس النموذج على نفس تقسيمات البيانات):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

في ML، يكون اختبار t المقترن شائعًا: حيث تقوم بتشغيل كلا النموذجين على نفس طيات التحقق المتبادل العشرة ومقارنة درجاتهما بشكل زوجي.

### Chi-squared Test

يتحقق اختبار مربع كاي مما إذا كانت الترددات المرصودة تتطابق مع الترددات المتوقعة. مفيدة للبيانات الفئوية.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for ML Models

اختبار A/B في ML ليس هو نفسه اختبار A/B على الويب. تنطوي مقارنة النماذج على تحديات محددة:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**الإجراء:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Statistical Significance vs Practical Significance

يمكن أن تكون النتيجة ذات دلالة إحصائية ولكنها لا معنى لها من الناحية العملية. ومع وجود بيانات كافية، يصبح حتى الفرق التافه ذو دلالة إحصائية.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**حجم التأثير** يحدد حجم الفرق، بغض النظر عن حجم العينة:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

قم دائمًا بالإبلاغ عن كل من القيمة p وحجم التأثير. تخبرك القيمة p إذا كان الفرق حقيقيًا. يخبرك حجم التأثير إذا كان الأمر مهمًا.

### Multiple Comparison Problem

عندما تختبر العديد من الفرضيات، سيكون بعضها "مهمًا" بالصدفة. إذا قمت باختبار 20 شيئًا عند ألفا = 0.05، فإنك تتوقع نتيجة إيجابية كاذبة واحدة حتى عندما لا يكون هناك شيء حقيقي.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**تصحيح بونفيروني:** قسمة ألفا على عدد الاختبارات.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

في ML، يكون هذا مهمًا عند مقارنة نموذج عبر مقاييس متعددة، أو اختبار العديد من تكوينات المعلمات الفائقة، أو التقييم على مجموعات بيانات متعددة.

### Bootstrap Methods

يقوم Bootstrapping بتقدير توزيع العينات للإحصائيات عن طريق إعادة تشكيل بياناتك مع الاستبدال. لا توجد افتراضات حول التوزيع الأساسي المطلوب.

**الخوارزمية:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**فاصل الثقة في Bootstrap (الطريقة المئوية):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

** لماذا يعتبر bootstrap مهمًا لـ ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap لمقارنة النماذج:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

هذا أقوى من اختبار t المقترن لأنه make لا يحتوي على افتراضات توزيعية.

### Parametric vs Non-parametric Tests

**الاختبارات البارامترية** تفترض توزيعًا محددًا (عاديًا عادةً):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**الاختبارات غير المعلمية** make لا توجد افتراضات توزيعية:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

** متى تستخدم غير المعلمية: **

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

** متى تستخدم المعلمة: **

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

في تجارب ML، عادةً ما يكون لديك عدد صغير من n (5 أو 10 طيات للتحقق المتبادل)، لذا فإن الاختبارات غير البارامترية مثل تصنيف ويلكوكسون غالبًا ما تكون أكثر ملاءمة من اختبارات t.

### Central Limit Theorem: Practical Implications

يشير CLT إلى أن توزيع العينة يقترب من التوزيع الطبيعي مع نمو n، بغض النظر عن التوزيع السكاني الأساسي.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

** لماذا هذا مهم لـ ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

** ماذا CLT يفعل NOT:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Common Statistical Mistakes in ML Papers

1. **الاختبار على مجموعة التدريب.** ضمانات التجهيز الزائد. احتفظ دائمًا بالبيانات التي لم يراها النموذج مطلقًا أثناء التدريب.

2. **لا توجد فترات ثقة.** الإبلاغ عن رقم دقة واحد دون عدم اليقين makes النتائج غير قابلة للتكرار ولا يمكن التحقق منها.

3. **تجاهل المقارنات المتعددة.** يؤدي اختبار 50 تكوينًا والإبلاغ عن أفضلها دون تصحيح إلى تضخيم المعدلات الإيجابية الخاطئة.

4. ** أهمية إحصائية وعملية مربكة. ** القيمة الاحتمالية البالغة 0.001 عند تحسين الدقة بنسبة 0.01% ليست ذات معنى.

5. **استخدام الدقة في البيانات غير المتوازنة.** الدقة بنسبة 99% في مجموعة بيانات ذات فئة سلبية بنسبة 99% تعني أن النموذج لم يتعلم شيئًا. استخدم الدقة أو التذكير أو F1 أو AUC.

6. **مقاييس الانتقاء.** الإبلاغ فقط عن المقياس الذي يفوز فيه نموذجك. تقارير التقييم الصادق عن جميع المقاييس ذات الصلة.

7. **تسرب المعلومات عبر تقسيمات التدريب/الاختبار.** التسوية قبل التقسيم، أو استخدام البيانات المستقبلية للتنبؤ بالماضي.

8. **مجموعات اختبار صغيرة بدون تقديرات تباين.** التقييم على 100 عينة والمطالبة بتحسن بنسبة 2% هو ضوضاء، وليس إشارة.

9. **افتراض الاستقلال عندما لا تكون البيانات مستقلة.** صور طبية من نفس المريض، جمل متعددة من نفس الوثيقة. ترتبط الملاحظات داخل المجموعة.

10. **P-hacking.** تجربة اختبارات أو مجموعات فرعية أو معايير استبعاد مختلفة حتى تحصل على P <0.05. والنتيجة هي قطعة أثرية من البحث.

## Building It

سوف تقوم بتنفيذ:

1. **الإحصائيات الوصفية من الصفر** (المتوسط، الوسيط، الوضع، الانحراف المعياري، النسب المئوية، IQR)
2. **دوال الارتباط** (بيرسون وسبيرمان، مع مصفوفة التغاير)
3. **اختبارات الفرضيات** (اختبار t لعينة واحدة، واختبار t لعينتين، واختبار مربع كاي)
4. **فترات الثقة في Bootstrap** (لأي إحصائية، لا حاجة إلى افتراضات)
5. **محاكي اختبار A/B** (إنشاء البيانات والاختبار والتحقق من أخطاء النوع الأول والنوع II)
6. **عرض توضيحي للأهمية الإحصائية مقابل الأهمية العملية** (يُظهر أن كل شيء "مهم" كبير)

كل ذلك من الصفر، باستخدام `math` و`random` فقط. لا numpy، لا scipy.

## Key Terms

| مصطلح | التعريف |
|---|---|
| يعني | مجموع القيم مقسوما على العدد. حساسة للقيم المتطرفة. |
| الوسيط | القيمة المتوسطة للبيانات التي تم فرزها. قوية للقيم المتطرفة. |
| الانحراف المعياري | الجذر التربيعي للتباين. التدابير المنتشرة في الوحدات الأصلية. |
| المئوي | القيمة التي تقل عنها نسبة معينة من البيانات. |
| IQR | النطاق الرباعي. Q3 ناقص Q1. انتشار الوسط 50%. |
| ارتباط بيرسون | يقيس الارتباط الخطي بين متغيرين. النطاق [-1، 1]. |
| ارتباط سبيرمان | يقيس الارتباط الرتيب باستخدام الرتب. |
| مصفوفة التغاير | مصفوفة التباينات الزوجية بين جميع الميزات. |
| الفرضية الصفرية | الافتراض الافتراضي عدم وجود تأثير أو عدم وجود فرق. |
| القيمة p | احتمال البيانات بهذا الحد الأقصى في ضوء الفرضية الصفرية صحيح. |
| فاصل الثقة | نطاق القيم المعقولة للمعلمة عند مستوى ثقة معين. |
| اختبار t | اختبارات ما إذا كانت الوسائل تختلف بشكل كبير. يستخدم توزيع t. |
| اختبار مربع كاي | يختبر ما إذا كانت الترددات المرصودة تختلف عن الترددات المتوقعة. |
| حجم التأثير | حجم الفرق، بغض النظر عن حجم العينة. كوهين د شائع. |
| تصحيح بونفيروني | يقسم عتبة الأهمية على عدد الاختبارات للتحكم في الإيجابيات الكاذبة. |
| بوتستراب | إعادة أخذ العينات مع الاستبدال لتقدير توزيعات العينات. |
| خطأ من النوع الأول | إيجابية كاذبة. رفض H0 عندما يكون صحيحا. |
| اكتب II خطأ | سلبية كاذبة. عدم الرفض H0 عندما يكون كاذباً. |
| القوة الإحصائية | احتمال الرفض الصحيح للخطأ H0. الطاقة = 1 ناقص النوع II معدل الخطأ. |
| نظرية الحد المركزي | تتقارب وسائل العينة مع التوزيع الطبيعي مع نمو حجم العينة. |
| اختبار بارامتري | يفترض توزيعًا محددًا للبيانات (عاديًا عادةً). |
| اختبار غير معلمي | لا يقدم أي افتراضات التوزيعية. يعمل على الرتب أو العلامات. |
