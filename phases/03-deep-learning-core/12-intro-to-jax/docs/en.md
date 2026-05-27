# Introduction to JAX

> PyTorch يحول الموترات. TensorFlow يبني الرسوم البيانية. JAX يجمع وظائف نقية. هذا الأخير يغير طريقة تفكيرك في التعلم العميق.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات:** المرحلة 03 الدروس 01-10، الأساسي NumPy
**الوقت:** ~90 دقيقة

## Learning Objectives

- اكتب رمز الشبكة العصبية ذي الوظيفة الخالصة باستخدام API الوظيفي لـ JAX (jax.numpy، jax.grad، jax.jit، jax.vmap)
- اشرح الفرق الرئيسي في التصميم بين طفرة PyTorch المتلهفة ونموذج التجميع الوظيفي لـ JAX
- تطبيق تجميع jit وتوجيه vmap لتسريع حلقات التدريب مقارنة بـ Python الساذجة
- تدريب شبكة بسيطة في JAX ومقارنة إدارة الحالة الصريحة مع نهج PyTorch الموجه للكائنات

## The Problem

أنت تعرف كيفية بناء الشبكات العصبية في PyTorch. يمكنك تحديد `nn.Module`، الاتصال بـ `.backward()`، خطوة المحسن. إنها تعمل. الملايين من الناس يستخدمونه.

لكن PyTorch لديه قيد مخبأ في DNA: فهو يتتبع العمليات بفارغ الصبر، واحدة تلو الأخرى، في بايثون. كل `tensor + tensor` هو إطلاق منفصل للنواة. كل خطوة تدريبية تعيد تفسير نفس كود بايثون. يعمل هذا بشكل جيد حتى تحتاج إلى تدريب نموذج يحتوي على 540 مليار معلمة عبر 2048 وحدة TPU. ثم يقتلك النفقات العامة.

يقوم Google DeepMind بتدريب الجوزاء على JAX. قام كلود بتدريب الأنثروبي على JAX. هذه ليست عمليات صغيرة - إنها أكبر عملية تدريب على الشبكة العصبية على وجه الأرض. لقد اختاروا JAX لأنه يتعامل مع حلقة التدريب الخاصة بك كبرنامج قابل للتجميع، وليس كسلسلة من مكالمات بايثون.

JAX هو NumPy بثلاث قوى خارقة: التمايز التلقائي، JIT التحويل البرمجي إلى XLA، والتوجيه التلقائي. تكتب دالة تعالج مثالاً واحدًا. JAX يمنحك وظيفة تقوم بمعالجة الدفعة، وحساب التدرجات، والتجميع إلى رمز الجهاز، وتشغيلها عبر أجهزة متعددة. كل ذلك دون تغيير الوظيفة الأصلية.

## The Concept

### The JAX Philosophy

JAX هو إطار وظيفي. لا توجد فئات، ولا حالة قابلة للتغيير، ولا توجد طريقة `.backward()`. بدلاً من:

| PyTorch | JAX |
|---------|-----|
| `nn.Module` فئة مع الولاية | وظيفة خالصة: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| تنفيذ حريص | JIT تجميع عبر XLA |
| `for x in batch:` حلقة يدوية | `jax.vmap(f)` التوجيه التلقائي |
| `DataParallel` / `FSDP` | `jax.pmap(f)` التوازي التلقائي |
| قابل للتغيير `model.parameters()` | pytree غير قابل للتغيير من المصفوفات |

هذا ليس تفضيلاً للأسلوب. إنه قيد المترجم. JIT يتطلب التجميع وظائف خالصة - تنتج نفس المدخلات دائمًا نفس المخرجات، دون أي آثار جانبية. هذا القيد هو make إمكانية تسريع 100x.

### jax.numpy: The Familiar Surface

JAX يعيد تنفيذ NumPy API على المسرعات:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

نفس أسماء الوظائف نفس قواعد البث نفس دلالات التقطيع. لكن المصفوفات تعيش على GPU/TPU، ويمكن تتبع كل عملية بواسطة المترجم.

فرق واحد حاسم: JAX المصفوفات غير قابلة للتغيير. لا `a[0] = 5`. بدلا من ذلك: `a = a.at[0].set(5)`. يبدو هذا أمرًا غريبًا لمدة أسبوع، ثم ينقر - الثبات هو ما يمكن تركيبه من تحويلات make مثل `grad` و`jit` و`vmap`.

### jax.grad: Functional Autodiff

PyTorch يربط التدرجات بالموترات (`.grad`). JAX إرفاق التدرجات للوظائف.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` يأخذ دالة ويعيد دالة جديدة تحسب التدرج. لا توجد مكالمة `.backward()`. لا يوجد رسم بياني حسابي مخزن على الموترات. التدرج هو مجرد وظيفة أخرى يمكنك استدعاؤها أو إنشائها أو تجميعها JIT.

هذا يتكون بشكل تعسفي:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

المشتقات الثانية. المشتقات الثالثة. اليعاقبة. هسه. كل ذلك من خلال تأليف `grad`. PyTorch يمكنه القيام بذلك أيضًا (`torch.autograd.functional.hessian`)، لكنه مثبت بمسامير. وفي JAX هو الأساس.

القيد: `grad` يعمل فقط على الوظائف النقية. لا توجد عبارات طباعة بالداخل (يتم تشغيلها أثناء التتبع، وليس التنفيذ). لا طفرة في الحالة الخارجية. لا يوجد توليد أرقام عشوائية دون إدارة مفاتيح واضحة.

### jit: Compile to XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

في المكالمة الأولى، JAX يتتبع الوظيفة - ويسجل العمليات التي تحدث، دون تنفيذها. ثم يتم تسليم هذا التتبع إلى XLA (الجبر الخطي المتسارع)، مترجم Google لـ TPUs وGPUs. XLA يدمج العمليات، ويزيل نسخ الذاكرة الزائدة عن الحاجة، ويولد رمز الجهاز الأمثل.

الاستدعاءات اللاحقة تتخطى لغة بايثون تمامًا. يتم تشغيل التعليمات البرمجية المترجمة على المسرع بسرعة C++.

عندما يساعد JIT:
- خطوات التدريب (نفس الحساب يتكرر آلاف المرات)
- الاستدلال (نفس النموذج، مدخلات مختلفة)
- أي دالة يتم استدعاؤها أكثر من مرة بمدخلات متشابهة الشكل

عندما JIT يؤلم:
- وظائف مع تدفق التحكم في Python الذي يعتمد على القيم (`if x > 0` حيث x عبارة عن مصفوفة متتبعة)
- حسابات طلقة واحدة (تتجاوز تكلفة التجميع وقت التشغيل)
- تصحيح الأخطاء (التتبع يخفي التنفيذ الفعلي)

تقييد تدفق التحكم حقيقي. `jax.lax.cond` يستبدل `if/else`. `jax.lax.scan` يستبدل `for` الحلقات. هذه ليست اختيارية - فهي ثمن التجميع.

### vmap: Automatic Vectorization

تكتب دالة تعالج مثالاً واحدًا:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` يرفعها لمعالجة دفعة:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` تعني: لا تقم بتجميع دفعة فوق `params` (مشتركة)، دفعة فوق المحور 0 من `x`. لا توجد حلقة يدوية `for`. لا إعادة تشكيل. لا يوجد خيوط البعد الدفعي. JAX يحدد أبعاد الدُفعة ويوجه الحساب بأكمله.

هذا ليس السكر النحوي. `vmap` ينشئ تعليمات برمجية موجهة مدمجة تعمل بمعدل 10-100x أسرع من حلقة Python. ويتكون من `jit` و `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

التدرجات لكل مثال. سطر واحد. هذا يكاد يكون مستحيلاً في PyTorch بدون اختراقات.

### pmap: Data Parallelism Across Devices

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` يكرر الوظيفة عبر جميع الأجهزة المتاحة (GPUs/وحدات TPU) ويقسم الدفعة. داخل الوظيفة، يقوم `jax.lax.pmean` و `jax.lax.psum` بمزامنة التدرجات عبر الأجهزة.

تقوم Google بتدريب Gemini على آلاف شرائح TPU v5e باستخدام `pmap` (والتي تليها `shard_map`). نموذج البرمجة: اكتب إصدار الجهاز الواحد، ثم اختتم بـ `pmap`، انتهى.

### Pytrees: The Universal Data Structure

JAX يعمل على "pytrees" - مجموعات متداخلة من القوائم، والصفوف، والإملاء، والمصفوفات. معلمات النموذج الخاص بك هي pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

كل تحويل JAX -- `grad`، `jit`، `vmap` -- يعرف كيفية اجتياز أشجار البيتري. `jax.tree.map(f, tree)` ينطبق `f` على كل ورقة. هذه هي الطريقة التي يقوم بها المحسنون بتحديث جميع المعلمات مرة واحدة:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

لا توجد طريقة `.parameters()`. لا يوجد تسجيل المعلمة. هيكل الشجرة هو النموذج.

### Functional vs Object-Oriented

PyTorch يخزن الحالة داخل الكائنات:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX يستخدم وظائف خالصة بحالة صريحة:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

يتم تمرير المعلمات. لا يتم تخزين أي شيء. لا شيء متحور. هذا make كل وظيفة قابلة للاختبار والتركيب والتجميع. ويعني ذلك أيضًا أنك تدير المعلمات بنفسك - أو تستخدم مكتبة مثل Flax أو Equinox.

### The JAX Ecosystem

JAX يمنحك البدائيات. المكتبات تمنحك بيئة العمل:

| مكتبة | الدور | النمط |
|---------|------|-------|
| **الكتان** (جوجل) | طبقات الشبكة العصبية | `nn.Module` بحالة صريحة |
| **إكوينوكس** (باتريك كيدجر) | طبقات الشبكة العصبية | مبني على Pytree، بايثونيك |
| **أوبتاكس** (ديب مايند) | محسنات + جداول LR | تحويلات التدرج المركبة |
| **أورباكس** (جوجل) | نقاط التفتيش | حفظ/استعادة pytrees |
| **CLU** (جوجل) | المقاييس + التسجيل | أدوات حلقة التدريب |

Optax هي مكتبة المحسن القياسية. إنه يفصل تحويل التدرج (آدم، SGD، القطع) عن تحديث المعلمة، مما يجعل تكوينه أمرًا تافهًا:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch

| عامل | JAX | PyTorch |
|--------|-----|---------|
| TPU الدعم | من الدرجة الأولى (قامت جوجل ببناء كليهما) | صيانة المجتمع (torch_xla) |
| GPU الدعم | جيد (CUDA عبر XLA) | الأفضل في فئته (أصلي CUDA) |
| التصحيح | الصعب (تتبع + تجميع) | سهل (حريص، سطرًا بسطر) |
| النظام البيئي | تركز على الأبحاث (الكتان، الاعتدال) | ضخمة (معانقة الوجه، torchvision، وما إلى ذلك) |
| توظيف | المتخصصة (جوجل / ديب مايند / أنثروبيك) | السائد (في كل مكان) |
| تدريب واسع النطاق | متفوق (XLA، pmap، مش) | جيد (FSDP، DeepSpeed) |
| سرعة النماذج الأولية | أبطأ (الحمل الوظيفي) | أسرع (تحور وانطلق) |
| استنتاج الإنتاج | خدمة TensorFlow، فيرتكس AI | تورش سيرف، تريتون، ONNX |
| من يستخدمه | ديب مايند (الجوزاء)، أنثروبي (كلود) | ميتا (اللاما)، OpenAI (GPT)، الاستقرار AI |

الإجابة الصادقة: استخدم PyTorch إلا إذا كان لديك سبب محدد لاستخدام JAX. هذه الأسباب هي - الوصول إلى TPU، أو الحاجة إلى التدرجات لكل مثال، أو التدريب على أجهزة متعددة على نطاق واسع، أو العمل في Google/DeepMind/Anthropic.

### Random Numbers in JAX

JAX ليس لديه حالة عشوائية عالمية. تتطلب كل عملية عشوائية مفتاحًا صريحًا PRNG:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

وهذا أمر مزعج في البداية. ولكنه يضمن إمكانية التكرار عبر الأجهزة والمجموعات - وهي خاصية لا يمكن لـ PyTorch's `torch.manual_seed` ضمانها في إعدادات GPU المتعددة.

## Build It

### Step 1: Setup and Data

سنقوم بتدريب 3 طبقات MLP على MNIST باستخدام JAX وOptax. 784 مدخلاً، طبقتان مخفيتان من 256 و128 خلية عصبية، 10 فئات مخرجات.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Step 2: Initialize Parameters

لا فئة. مجرد وظيفة تقوم بإرجاع pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

تتم التهيئة يدويًا. ثلاثة PRNG مفاتيح مقسمة من بذرة واحدة. كل وزن عبارة عن مصفوفة غير قابلة للتغيير في إملاء متداخل.

### Step 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

وظائف نقية. المعلمات في الداخل والتنبؤ بالخارج. لا `self`، لا توجد حالة مخزنة. `loss_fn` يحسب الإنتروبيا المتقاطعة من الصفر - softmax، log، Negative Mean.

### Step 4: JIT-Compiled Training Step

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` تُرجع كلاً من قيمة الخسارة والتدرجات في مسار واحد. يقوم مصمم الديكور `@jax.jit` بتجميع كلتا الوظيفتين إلى XLA. بعد المكالمة الأولى، يتم تشغيل كل خطوة تدريب دون لمس بايثون.

### Step 5: Training Loop

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 عصور. ~97% دقة الاختبار. العصر الأول بطيء (JIT تجميع). العصور 2-10 سريعة.

لاحظ ما هو مفقود: لا `.zero_grad()`، لا `.backward()`، لا `.step()`. التحديث بأكمله عبارة عن استدعاء دالة مؤلف واحد. يتم حساب التدرجات، وتحويلها بواسطة Adam، وتطبيقها على المعلمات - كل ذلك داخل `train_step`.

## Use It

### Flax: The Google Standard

الكتان هو مكتبة الشبكات العصبية الأكثر شيوعًا JAX. يضيف `nn.Module` مرة أخرى، ولكن مع إدارة الحالة الصريحة:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

نفس بنية PyTorch، لكن `params` منفصلة عن النموذج. `model.init()` ينشئ المعلمات. `model.apply(params, x)` يدير التمريرة الأمامية. كائن النموذج ليس له حالة.

### Equinox: The Pythonic Alternative

يمثل Equinox (بواسطة باتريك كيدجر) نماذج مثل أشجار البيريت:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

النموذج نفسه هو pytree. لا حاجة إلى `.apply()`. المعلمات هي مجرد أوراق النموذج. هذا أقرب إلى كيفية تفكير JAX.

### Optax: Composable Optimizers

يقوم Optax بفصل تحويل التدرج عن التحديث:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

قص متدرج، إحماء معدل التعلم، تناقص الوزن - كل ذلك يتكون في سلسلة من التحولات. يرى كل تحويل التدرجات، ويعدلها، ويمررها إلى التالي. لا توجد فئة محسن متجانسة.

## Ship It

**Installation:**

```bash
pip install jax jaxlib optax flax
```

للحصول على دعم GPU:

```bash
pip install jax[cuda12]
```

بالنسبة إلى TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

** مسك الأداء: **

- المكالمة JIT الأولى بطيئة (تجميع). الاحماء قبل قياس الأداء.
- تجنب حلقات Python فوق المصفوفات JAX داخل JIT. استخدم `jax.lax.scan` أو `jax.lax.fori_loop`.
- `jax.debug.print()` يعمل داخل JIT. العادية `print()` لا.
- الملف الشخصي مع `jax.profiler` أو TensorBoard. XLA التجميع يمكن أن يخفي الاختناقات.
- JAX يخصص مسبقًا 75% من GPU الذاكرة بشكل افتراضي. اضبط `XLA_PYTHON_CLIENTENT_PREALLOCATE=false` للتعطيل.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**ينتج من هذا الدرس:**
- `outputs/prompt-jax-optimizer.md` -- مطالبة باختيار التكوين المحسّن JAX المناسب
- `outputs/skill-jax-patterns.md` -- مهارة تغطي الأنماط الوظيفية في JAX

## Exercises

1. أضف التسرب إلى MLP. في JAX، يتطلب التسرب مفتاح PRNG - قم بتمرير المفتاح من خلال التمريرة الأمامية وتقسيمه لكل طبقة منسدلة. قارن دقة الاختبار مع وبدون.

2. استخدم `jax.vmap` لحساب التدرجات لكل مثال لمجموعة مكونة من 32 MNIST صورة. حساب قاعدة التدرج لكل مثال. ما هي الأمثلة التي تحتوي على أكبر التدرجات، ولماذا؟

3. استبدل وظيفة التوجيه اليدوية بوظيفة `mlp_forward(params, x)` عامة تعمل مع أي عدد من الطبقات. استخدم `jax.tree.leaves` لتحديد العمق تلقائيًا.

4. قم بمقارنة خطوة التدريب مع وبدون `@jax.jit`. الوقت 100 خطوة لكل منهما. ما هو حجم التسريع على أجهزتك؟ ما هي النفقات العامة للتجميع في المكالمة الأولى؟

5. قم بتنفيذ القص المتدرج من خلال إنشاء `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. تدريب مع وبدون قص. ارسم معيار التدرج على التدريب لرؤية التأثير.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| XLA | "الشيء الذي makeس JAX سريع" | الجبر الخطي المتسارع - مترجم يدمج العمليات ويولد نواة GPU/TPU محسنة من رسم بياني حسابي |
| JIT | "التجميع في الوقت المناسب" | JAX يتتبع الوظيفة عند الاتصال الأول، ويترجم إلى XLA، ثم يقوم بتشغيل النسخة المترجمة على المكالمات اللاحقة |
| وظيفة نقية | "ليس له آثار جانبية" | دالة يعتمد فيها الإخراج فقط على المدخلات - لا توجد حالة عالمية، ولا طفرة، ولا عشوائية بدون مفاتيح صريحة |
| vmap | "التجميع التلقائي" | يحول دالة تقوم بمعالجة مثال واحد إلى دالة تعالج دفعة، دون إعادة كتابة |
| بي ماب | "التوازي التلقائي" | ينسخ وظيفة عبر أجهزة متعددة ويقسم دفعة الإدخال |
| بيتري | "إملاء المصفوفات المتداخلة" | أي بنية متداخلة من القوائم والصفوف والإملاء والمصفوفات التي يمكن لـ JAX اجتيازها وتحويلها |
| تتبع | "تسجيل الحساب" | JAX ينفذ الدالة بقيم مجردة لبناء رسم بياني حسابي، دون حساب النتائج الحقيقية |
| تمييز تلقائي وظيفي | "درجة الدالة" | حساب المشتقات عن طريق تحويل الدوال، وليس عن طريق ربط مخزن التدرج بالموترات |
| اوبتاكس | "مكتبة مُحسّنات JAX" | مكتبة قابلة للتركيب من تحويلات التدرج - آدم، SGD، القطع، الجدولة - تلك السلسلة معًا |
| الكتان | "JAX's nn.Module" | مكتبة الشبكة العصبية من Google لـ JAX، إضافة تجريدات الطبقة مع الحفاظ على الحالة الصريحة |

## Further Reading

- JAX التوثيق: https://jax.readthedocs.io/ -- المستندات الرسمية، مع برامج تعليمية ممتازة حول grad وjit وvmap
- "JAX: التحولات المركبة لبرامج Python+NumPy" (Bradbury et al., 2018) - الورقة الأصلية التي تشرح فلسفة التصميم
- وثائق الكتان: https://flax.readthedocs.io/ - مكتبة الشبكة العصبية من Google لـ JAX
- باتريك كيدجر، "Equinox: الشبكات العصبية في JAX عبر PyTrees القابلة للاستدعاء والتحويلات المصفاة" (2021) - البديل Pythonic لـ Flax
- DeepMind، "Optax: تحويل التدرج القابل للتركيب والتحسين" - مكتبة المحسن القياسية
- "أنت لا تعرف JAX" (كولين رافيل، 2020) - دليل عملي لـ JAX مسكتك وأنماطها، من أحد مؤلفي T5
