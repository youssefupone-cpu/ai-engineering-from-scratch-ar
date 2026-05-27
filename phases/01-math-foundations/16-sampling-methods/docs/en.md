# Sampling Methods

> أخذ العينات هو كيفية استكشاف AI لمساحة الاحتمالات.

**النوع:** بناء
** اللغة: ** بايثون
**المتطلبات الأساسية:** المرحلة الأولى، الدروس 06-07 (الاحتمالات، نظرية بايز)
**الوقت:** ~120 دقيقة

## Learning Objectives

- تنفيذ أخذ العينات العكسية CDF والرفض والأهمية من الصفر باستخدام أرقام عشوائية موحدة فقط
- إنشاء عينات لدرجة الحرارة وtop-k وtop-p (النواة) لإنشاء الرمز المميز لنموذج اللغة
- شرح خدعة إعادة المعلمة ولماذا تتيح الانتشار العكسي من خلال أخذ العينات في VAEs
- قم بتشغيل Metropolis-Hastings MCMC لأخذ عينة من التوزيع المستهدف غير الطبيعي

## The Problem

ينتهي نموذج اللغة من معالجة موجهك وينتج متجهًا قدره 50000 logits. واحد لكل رمز في مفرداته. الآن عليها أن تختار واحدة. كيف؟

إذا اختار دائمًا الرمز المميز ذو الاحتمالية الأعلى، فستكون كل استجابة متطابقة. حتمية. ممل. إذا تم اختياره بشكل عشوائي بشكل موحد، يكون الإخراج رطانة. الجواب يكمن في مكان ما بين هذين النقيضين، ويتم التحكم في هذا المكان عن طريق أخذ العينات.

لا يقتصر أخذ العينات على إنشاء النص. يقدر التعلم المعزز تدرجات السياسة عن طريق أخذ عينات من المسارات. تتعلم VAEs التمثيلات الكامنة عن طريق أخذ عينات من التوزيعات المستفادة والانتشار العكسي من خلال العشوائية. تقوم نماذج الانتشار بإنشاء صور عن طريق أخذ عينات من الضوضاء وتقليل الضوضاء بشكل متكرر. تقوم طرق مونت كارلو بتقدير التكاملات التي ليس لها حل مغلق. MCMC تستكشف الخوارزميات التوزيعات الخلفية عالية الأبعاد التي يستحيل تعدادها.

كل نظام AI توليدي هو نظام أخذ العينات. تحدد استراتيجية أخذ العينات جودة المخرجات وتنوعها وإمكانية التحكم فيها. يبني هذا الدرس كل طرق أخذ العينات الرئيسية من الصفر، بدءًا من الأرقام العشوائية الموحدة وانتهاءً بالتقنيات التي تدعم النماذج LLMs الحديثة والنماذج التوليدية.

## The Concept

### Why Sampling Matters

يظهر أخذ العينات في أربعة أدوار أساسية عبر AI والتعلم الآلي:

**الجيل.** تنتج نماذج اللغة ونماذج الانتشار وشبكات GAN مخرجات عن طريق أخذ العينات. تتحكم خوارزمية أخذ العينات بشكل مباشر في الإبداع والتماسك والتنوع. درجة الحرارة، وأعلى مستوى، وأخذ عينات النواة هي المقابض التي يديرها المهندسون يوميًا.

**التدريب.** عينات النسب التدرج العشوائي على دفعات صغيرة. عينات التسرب من الخلايا العصبية لإلغاء تنشيطها. زيادة البيانات عينات التحولات العشوائية. تعمل عينات الأهمية على إعادة وزن العينات لتقليل تباين التدرج في التعلم المعزز (PPO، TRPO).

**التقدير.** العديد من الكميات في ML ليس لها حل مغلق. الخسارة المتوقعة في توزيع البيانات، وظيفة التقسيم لنموذج قائم على الطاقة، الدليل في الاستدلال البايزي. ويقارب تقدير مونت كارلو كل هذه الأمور عن طريق حساب المتوسط ​​على العينات.

**الاستكشاف.** تستكشف خوارزميات MCMC التوزيعات الخلفية في الاستدلال البايزي. الاستراتيجيات التطورية عينة من اضطرابات المعلمة. يوازن أخذ عينات طومسون بين الاستكشاف والاستغلال في قطاع الطرق.

التحدي الأساسي: يمكنك فقط أخذ عينات مباشرة من التوزيعات البسيطة (الموحدة والعادية). بالنسبة لكل شيء آخر، تحتاج إلى طريقة لتحويل العينات البسيطة إلى عينات من التوزيع المستهدف.

### Uniform Random Sampling

كل طريقة لأخذ العينات تبدأ هنا. ينتج مولد الأرقام العشوائية الموحدة قيمًا في [0، 1) حيث يكون لكل فترة فرعية متساوية الطول احتمالية متساوية.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

لأخذ عينات بشكل موحد من مجموعة منفصلة من عناصر n، قم بإنشاء U وإرجاع Floor(n * U). لأخذ عينة من نطاق مستمر [a, b]، احسب a + (b - a) * U.

الفكرة الأساسية: يحتوي رقم عشوائي موحد على المقدار الصحيح من العشوائية لإنتاج عينة واحدة من أي توزيع. الحيلة هي إيجاد التحول الصحيح.

### Inverse CDF Method (Inverse Transform Sampling)

تقوم دالة التوزيع التراكمي (CDF) بتعيين القيم للاحتمالات:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

معكوس CDF يعيد الاحتمالات إلى القيم. إذا كان U ~ منتظم(0, 1)، فإن X = F_inverse(U) يتبع التوزيع المستهدف.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**مثال التوزيع الأسي:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

يعمل هذا بشكل مثالي عندما يمكنك كتابة F_inverse في شكل مغلق. بالنسبة للتوزيع الطبيعي، لا يوجد معكوس مغلق CDF، لذلك نستخدم طرق أخرى (بوكس مولر، أو التقريب العددي).

**إصدار منفصل:** بالنسبة إلى التوزيعات المنفصلة، ​​أنشئ CDF كمجموع تراكمي، وقم بإنشاء U، وابحث عن المؤشر الأول الذي يتجاوز فيه المجموع التراكمي U. هذه هي الطريقة التي يعمل بها `sample_categorical` في الدرس 06.

### Rejection Sampling

عندما لا تتمكن من عكس CDF ولكن يمكنك تقييم الهدف PDF حتى ثابت، فإن أخذ عينات الرفض يعمل.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

كلما كان الحد M أكثر إحكاما، كلما ارتفع معدل القبول. في الأبعاد المنخفضة (1-3)، تعمل عينات الرفض بشكل جيد. في الأبعاد العالية، ينخفض ​​معدل القبول بشكل كبير بسبب رفض معظم حجم الاقتراح. هذه هي لعنة الأبعاد لأخذ عينات الرفض.

**مثال: أخذ عينات من المستوى الطبيعي المقطوع.** استخدم اقتراحًا موحدًا على النطاق المقطوع. المغلف M هو الحد الأقصى للطبيعي PDF في هذا النطاق.

**مثال: أخذ العينات من نصف دائرة.** اقترح بشكل موحد في المستطيل المحيط. اقبل إذا كانت النقطة تقع داخل نصف الدائرة. هذه هي الطريقة التي تحسب بها مونتي كارلو pi: معدل القبول يساوي نسبة المساحة pi/4.

### Importance Sampling

في بعض الأحيان لا تحتاج إلى عينات من التوزيع المستهدف p(x). تحتاج إلى تقدير التوقع تحت p(x)، ولديك عينات من توزيع مختلف q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

وهذا أمر بالغ الأهمية في تعزيز التعلم. في PPO (تحسين السياسة القريبة)، تقوم بجمع المسارات ضمن سياسة قديمة pi_old ولكنك ترغب في تحسين سياسة جديدة pi_new. وزن الأهمية هو pi_new(a|s) / pi_old(a|s). PPO قص هذه الأوزان لمنع السياسة الجديدة من الابتعاد كثيرًا عن السياسة القديمة.

يعتمد تباين مقدر أخذ العينات على مدى تشابه q مع p. إذا كانت q مختلفة تمامًا عن p، فإن بعض العينات تحصل على أوزان هائلة وتهيمن على التقدير. يتم تقسيم أخذ عينات الأهمية ذاتية التطبيع على مجموع الأوزان لتقليل هذه المشكلة:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

تقدير مونت كارلو يقارب التكاملات عن طريق حساب متوسط ​​العينات العشوائية. قانون الأعداد الكبيرة يضمن التقارب.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

معدل الخطأ مستقل عن البعد. ولهذا السبب تهيمن أساليب مونت كارلو على الأبعاد العالية حيث يكون التكامل القائم على الشبكة مستحيلاً.

**تقدير باي:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**تقدير التوقعات:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC يبني سلسلة ماركوف التي يكون توزيعها الثابت هو التوزيع المستهدف p(x). وبعد خطوات كافية، تصبح العينات من السلسلة (تقريبًا) عينات من p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

بالنسبة للمقترحات المتماثلة (q(x'|x) = q(x|x'))، يتم تبسيط النسبة إلى p(x')/p(x). هذه هي خوارزمية متروبوليس الأصلية.

**سبب نجاحها.** تضمن قاعدة القبول توازنًا تفصيليًا: احتمالية الوجود عند x والانتقال إلى x' تساوي احتمالية الوجود عند x' والانتقال إلى x. يشير التوازن التفصيلي إلى أن p(x) هو التوزيع الثابت للسلسلة.

**اعتبارات عملية:**
- الاحتراق: تجاهل العينات المبكرة قبل أن تصل السلسلة إلى التوازن
- التخفيف: احتفظ بكل عينة من النوع k لتقليل الارتباط الذاتي
- نطاق الاقتراح: صغير جدًا وتتحرك السلسلة ببطء (قبول مرتفع، استكشاف بطيء)؛ كبيرة جدًا ويتم رفض معظم المقترحات (قبول منخفض، عالقة في مكانها)
- معدل القبول الأمثل للمقترح الغاوسي في الأبعاد العالية هو 0.234 تقريبًا

### Gibbs Sampling

أخذ عينات Gibbs هو حالة خاصة من MCMC للتوزيعات متعددة المتغيرات. فبدلاً من اقتراح نقلة في جميع الأبعاد مرة واحدة، فإنها تقوم بتحديث متغير واحد في كل مرة من توزيعها المشروط.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

يتطلب أخذ عينات Gibbs أنه يمكنك أخذ عينة من كل توزيع شرطي p(x_i | x_{-i}). وهذا واضح بالنسبة للعديد من النماذج:
- الشبكات البايزية: الشروط الشرطية تتبع بنية الرسم البياني
- الخلائط الغوسية: الشرطية غاوسية
- نماذج Ising: يعتمد كل دوران مشروط على جيرانه فقط

معدل القبول هو دائمًا 1 (يتم قبول كل اقتراح) لأن أخذ العينات من الشرط الشرطي الدقيق يلبي تلقائيًا الرصيد التفصيلي.

**القيود.** عندما تكون المتغيرات مترابطة بشكل كبير، يتم خلط عينات Gibbs ببطء لأن تحديث متغير واحد في كل مرة لا يمكن make تحركات قطرية كبيرة عبر التوزيع.

### Temperature Sampling (Used in LLMs)

نماذج اللغة تخرج logits z_1،...، z_V لكل رمز مميز في المفردات. يقوم Softmax بتحويل هذه إلى احتمالات. تقوم درجة الحرارة بإعادة قياس logits قبل softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**لماذا يعمل.** تؤدي قسمة logits على T < 1 إلى تضخيم الاختلافات بين logits. إذا كانت z_1 = 2 وz_2 = 1، فإن القسمة على T = 0.5 تعطي z_1/T = 4 وz_2/T = 2، مما يجعل الفجوة أكبر. بعد softmax، يحصل الرمز المميز الأعلى logit على حصة أكبر بكثير.

** في الممارسة العملية: **
- T = 0.0: فك التشفير الجشع، الأفضل للأسئلة والأجوبة الواقعية
- T = 0.3-0.7: إبداعي قليلًا، وجيد لإنشاء التعليمات البرمجية
- T = 0.7-1.0: متوازن وجيد للمحادثة العامة
- T = 1.0-1.5: الكتابة الإبداعية، العصف الذهني
- T> 1.5: عشوائي بشكل متزايد، ونادرا ما يكون مفيدا

درجة الحرارة لا تغير الرموز الممكنة. يغير كتلة الاحتمالية المخصصة لكل رمز.

### Top-k Sampling

يقوم أخذ عينات Top-k بتقييد تعيين المرشح على الرموز المميزة k ذات أعلى الاحتمالات، ثم إعادة التطبيع وأخذ العينات من تلك المجموعة المقيدة.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

يمنع Top-k النموذج من اختيار الرموز المميزة غير المحتملة للغاية (الأخطاء المطبعية والهراء) الموجودة في الذيل الطويل لتوزيع المفردات. المشكلة: تم إصلاح k بغض النظر عن السياق. عندما يكون النموذج واثقًا (رمز واحد لديه احتمال 95٪)، فإن k = 40 لا يزال يسمح بـ 39 بديلاً. عندما يكون النموذج غير مؤكد (تنتشر الاحتمالية عبر 1000 رمز)، فإن k = 40 تقطع الخيارات المعقولة.

### Top-p (Nucleus) Sampling

يقوم أخذ العينات من Top-p بضبط حجم المجموعة المرشحة ديناميكيًا. بدلاً من الاحتفاظ بعدد ثابت من الرموز المميزة، فإنه يحتفظ بأصغر مجموعة من الرموز المميزة التي يتجاوز احتمالها التراكمي p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

عندما يكون النموذج واثقًا، فإن أخذ العينات النواة يحتفظ بعدد قليل من الرموز (ربما 2-3). عندما يكون النموذج غير مؤكد، فإنه يحتفظ بالكثير (ربما 200). هذا السلوك التكيفي هو السبب في أن أخذ العينات النووية ينتج بشكل عام نصًا أفضل من النص العلوي.

** المجموعات المشتركة: **
- درجة الحرارة 0.7 + أعلى درجة 0.9: إعداد جيد للأغراض العامة
- درجة الحرارة 0.0 (الجشع): الأفضل للمهام الحتمية
- درجة الحرارة 1.0 + أعلى ك 50: فان وآخرون. (2018) إعداد الورق الأصلي

يمكن الجمع بين Top-k وtop-p. قم بتطبيق top-k أولاً، ثم top-p على المجموعة المتبقية.

### Reparameterization Trick (Used in VAEs)

تتعلم أجهزة التشفير التلقائي المتغيرة (VAEs) عن طريق تشفير المدخلات في توزيع في الفضاء الكامن، وأخذ عينات من هذا التوزيع، وفك تشفير العينة مرة أخرى. المشكلة: لا يمكنك الانتشار العكسي من خلال عملية أخذ العينات.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

تفصل خدعة إعادة المعلمة العشوائية عن المعلمات:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

يعمل هذا لأن N(mu, sigma^2) له نفس التوزيع مثل mu + sigma * N(0, 1). الفكرة الأساسية: نقل العشوائية إلى مصدر خالٍ من المعلمات (epsilon)، ثم التعبير عن العينة كتحويل قابل للتمييز للمعلمات.

**في حلقة التدريب VAE:**
1. يقوم جهاز التشفير بإخراج mu وlog(sigma^2) لكل إدخال
2. عينة إبسيلون ~ N(0، 1)
3. حساب z = mu + سيجما * إبسيلون
4. قم بفك تشفير z لإعادة بناء الإدخال
5. الانتشار العكسي خلال الخطوات 4، 3، 2، 1 (ممكن لأن الخطوة 3 قابلة للتمييز)

بدون خدعة إعادة المعلمة، لا يمكن تدريب VAEs باستخدام الانتشار العكسي القياسي. هذه الرؤية الوحيدة جعلت VAEs عملية.

### Gumbel-Softmax (Differentiable Categorical Sampling)

تعمل خدعة إعادة المعلمة على التوزيعات المستمرة (الغاوسية). بالنسبة للتوزيعات الفئوية المنفصلة، ​​نحتاج إلى نهج مختلف. يوفر Gumbel-Softmax تقريبًا تفاضليًا لأخذ العينات الفئوية.

**خدعة غامبل-ماكس (غير قابلة للتمييز):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**غامبل-سوفت ماكس (تقريب متباين):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

ينتج Gumbel-Softmax استرخاءً مستمرًا لعينة منفصلة. الإخراج هو متجه احتمالي (ناعم-ساخن) بدلاً من متجه واحد-ساخن. تتدفق التدرجات من خلال softmax. أثناء التمريرة الأمامية في التدريب، يمكنك استخدام مقدر "المباشرة": استخدم تدرجات argmax الصلبة للتمريرة الأمامية ولكن تدرجات Gumbel-Softmax الناعمة للتمريرة الخلفية.

**التطبيقات:**
- المتغيرات الكامنة المنفصلة في VAEs
- البحث في الهندسة العصبية (اختيار العمليات المنفصلة)
- آليات الانتباه الصعب
- تعزيز التعلم من خلال إجراءات منفصلة

### Stratified Sampling

يمكن لعينة مونت كارلو القياسية أن تترك فجوات في مساحة العينة بالصدفة. إن أخذ العينات الطبقية يفرض التغطية بالتساوي عن طريق تقسيم المساحة إلى طبقات وأخذ عينات من كل منها.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

دائمًا ما يكون لأخذ العينات الطبقية تباين أقل أو متساوٍ مقارنةً بمونت كارلو القياسية:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**التطبيقات:**
- التكامل العددي (شبه مونت كارلو)
- تقسيم بيانات التدريب (ضمان توازن الفصل في كل حظيرة)
- أهمية أخذ العينات مع التقسيم الطبقي (الجمع بين كلا التقنيتين)
- يستخدم NeRF (حقول الإشعاع العصبي) أخذ عينات طبقية على طول أشعة الكاميرا

### Connection to Diffusion Models

تقوم نماذج الانتشار بتوليد الصور من خلال عملية أخذ العينات. تضيف العملية الأمامية ضوضاء غاوسية إلى الصورة عبر خطوات T حتى تصبح ضوضاء نقية. تتعلم العملية العكسية تقليل الضوضاء واستعادة الصورة الأصلية خطوة بخطوة.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

العلاقة بالطرق الموجودة في هذا الدرس:
- تستخدم كل خطوة لتقليل الضوضاء خدعة إعادة المعلمة (ضوضاء العينة، تطبيق التحويل الحتمي)
- يتحكم جدول الضوضاء {alpha_t} في شكل من أشكال التلدين بدرجة الحرارة
- يستخدم التدريب تقدير مونت كارلو لتقريب ELBO (الحد الأدنى للدليل)
- أخذ العينات السلفية في نماذج الانتشار هو سلسلة ماركوف (كل خطوة تعتمد فقط على الحالة الحالية)

إن عملية توليد الصورة بأكملها عبارة عن أخذ عينات تكرارية: ابدأ من الضوضاء، وفي كل خطوة، قم بعينة نسخة أقل ضوضاءً قليلاً مشروطة بنموذج تقليل الضوضاء الذي تم تعلمه.

## Build It

### Step 1: Uniform and inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

إنشاء 10000 عينة أسية والتحقق من أن المتوسط ​​هو 1/لامدا.

### Step 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

استخدم أخذ عينات الرفض للرسم من التوزيع الطبيعي المقطوع. التحقق من الشكل عن طريق الرسم البياني للعينات.

### Step 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

قم بتقدير E[X^2] ضمن التوزيع الطبيعي باستخدام اقتراح موحد. قارن بالإجابة المعروفة (mu^2 + sigma^2).

### Step 4: Monte Carlo estimation of pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Step 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

عينة من توزيع ثنائي النسق (خليط من اثنين من Gaussians). تصور مسار السلسلة.

### Step 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Step 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

أظهر كيف تغير درجة الحرارة توزيع المخرجات لمجموعة من الرموز المميزة logits.

### Step 8: Top-k and top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Step 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

إثبات أن التدرجات تتدفق من خلال العينة المعاد معاملتها ولكن ليس من خلال أخذ العينات المباشرة.

### Step 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

أظهر كيف أن انخفاض درجة الحرارة make يقترب من الخرج لمتجه واحد ساخن.

توجد عمليات التنفيذ الكاملة مع كافة المرئيات في `code/sampling.py`.

## Use It

مع NumPy وSciPy، إصدارات الإنتاج:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

بالنسبة لـ MCMC على نطاق واسع، استخدم المكتبات المخصصة:
- PyMC: النمذجة الافتراضية الكاملة مع NUTS (التكيف HMC)
- المدير: فرقة MCMC العينات
- NumPyro/JAX: GPU-متسارع MCMC

لقد بنيت هذه من الصفر. الآن أنت تعرف ما تفعله مكالمات المكتبة.

## Exercises

1. تنفيذ أخذ العينات العكسية CDF لتوزيع كوشي. CDF هو F(x) = 0.5 + arctan(x)/pi. قم بإنشاء 10000 عينة ورسم الرسم البياني مقابل PDF الحقيقي. لاحظ الذيول الثقيلة (القيم المتطرفة بعيدة عن المركز).

2. استخدم أخذ عينات الرفض لإنشاء عينات من توزيع بيتا (2، 5) باستخدام اقتراح موحد (0، 1). ارسم العينات المقبولة مقابل النسخة التجريبية الحقيقية PDF. ما هو معدل القبول النظري؟

3. قم بتقدير تكامل sin(x) من 0 إلى pi باستخدام مونت كارلو مع 1000 و10000 و100000 عينة. قارن الخطأ في كل مستوى. تحقق من أن مقياس الخطأ هو O(1/sqrt(N)).

4. قم بتطبيق Metropolis-Hastings لأخذ عينة من توزيع ثنائي الأبعاد p(x, y) متناسب مع exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). رسم العينات ومسار السلسلة. قم بتجربة الانحرافات المعيارية المختلفة للاقتراح.

5. أنشئ عرضًا توضيحيًا كاملاً لإنشاء النص: بالنظر إلى مفردات مكونة من 10 كلمات مع logits، قم بإنشاء تسلسلات من 20 رمزًا باستخدام (أ) الجشع، (ب) درجة الحرارة=0.7، (ج) top-k=3، (د) top-p=0.9. قارن تنوع النواتج عبر 5 أشواط.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| أخذ العينات | "رسم قيم عشوائية" | توليد القيم وفقا للتوزيع الاحتمالي. الآلية وراء كل توليدي AI |
| توزيع موحد | "الكل متساوٍ في الاحتمال" | كل قيمة في [a, b] لها كثافة احتمالية متساوية 1/(b-a). نقطة البداية لجميع طرق أخذ العينات |
| معكوس CDF | "التحويل الاحتمالي" | يقوم F_inverse(U) بتحويل عينة موحدة إلى عينة من أي توزيع معروف CDF. دقيق وفعال |
| أخذ عينات الرفض | "اقتراح وقبول/رفض" | أنشئ من اقتراح بسيط، واقبله باحتمال يتناسب مع نسبة الهدف/الاقتراح. دقيق ولكن يضيع العينات |
| أهمية أخذ العينات | "إعادة وزن العينات" | تقدير التوقعات تحت p(x) باستخدام عينات من q(x) عن طريق ترجيح كل عينة بواسطة p(x)/q(x). الأساسية إلى PPO في RL |
| مونت كارلو | "متوسط ​​العينات العشوائية" | التكاملات التقريبية كمتوسطات عينة. خطأ O(1/sqrt(N)) بغض النظر عن البعد |
| MCMC | "المشي العشوائي المتقارب" | قم ببناء سلسلة ماركوف التي يكون توزيعها الثابت هو الهدف. متروبوليس-هاستينغز هي الخوارزمية التأسيسية |
| متروبوليس هاستينغز | "اقبل صعودًا، وأحيانًا هبوطًا" | اقتراح التحركات، وقبول على أساس نسبة الكثافة. التوازن التفصيلي يضمن التقارب مع التوزيع المستهدف |
| أخذ عينات جيبس ​​| "متغير واحد في كل مرة" | قم بتحديث كل متغير من توزيعه المشروط مع تثبيت المتغيرات الأخرى. نسبة القبول 100% |
| درجة الحرارة | "مقبض الثقة" | يقسم logits على T قبل softmax. T<1 يزيد حدة (أكثر ثقة)، T>1 يسطح (أكثر تنوعًا) |
| أخذ العينات من أعلى ك | "حافظ على الأفضل" | قم بإزالة جميع الرموز المميزة ذات الاحتمالية الأعلى باستثناء k، ثم قم بإعادة التطبيع، ثم أخذ عينة. حجم مجموعة المرشحين الثابتة |
| أخذ عينات النواة (أعلى ع) | "احتفظ بالاحتمالات" | احتفظ بأصغر مجموعة من الرموز المميزة التي يتجاوز احتمالها التراكمي p. حجم مجموعة المرشح التكيفي |
| خدعة إعادة المعلمة | "حرك العشوائية للخارج" | اكتب z = mu + sigma * epsilon حيث epsilon ~ N(0,1). يجعل أخذ العينات قابلة للتمييز. ضروري للتدريب VAE |
| غامبل-سوفت ماكس | "أخذ العينات الفئوية الناعمة" | التقريب التفاضلي لأخذ العينات الفئوية باستخدام ضوضاء غامبل + softmax مع درجة الحرارة |
| أخذ العينات الطبقية | "التغطية القسرية" | قسّم مساحة العينة إلى طبقات، وعينة من كل منها. دائما أقل تباينا من مونت كارلو الساذجة |
| حرق في | "فترة الاحماء" | يتم التخلص من MCMC عينات أولية قبل أن تصل السلسلة إلى توزيعها الثابت |
| الرصيد التفصيلي | "شرط الرجوع" | ع(x) * T(x->y) = p(y) * T(y->x). الشرط الكافي لـ p هو التوزيع الثابت لسلسلة ماركوف |
| أخذ العينات من الانتشار | "التقليل التكراري" | إنشاء البيانات عن طريق البدء من الضوضاء وتطبيق خطوات تقليل الضوضاء المستفادة. كل خطوة هي عملية أخذ عينات مشروطة |

## Further Reading

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - detailed tutorial on MCMC foundations
- [Jang, Gu, Poole (2017): إعادة المعلمة الفئوية باستخدام Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - ورق Gumbel-Softmax الأصلي
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) ورق أخذ العينات
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE paper introducing the reparameterization trick
- [Ho, Jain, Abbeel (2020): تقليل الضوضاء النماذج الاحتمالية للانتشار](https://arxiv.org/abs/2006.11239) - DDPM يربط أخذ العينات بتوليد الصور
