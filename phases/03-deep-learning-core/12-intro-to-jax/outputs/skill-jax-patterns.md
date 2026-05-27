---
name: skill-jax-patterns
description: Functional programming patterns in JAX -- when and how to use grad, jit, vmap, and pmap
version: 1.0.0
phase: 3
lesson: 12
tags: [jax, functional-programming, autodiff, compilation, vectorization]
---

# JAX Functional Patterns

JAX يحول الوظائف النقية. يتبع كل نمط أدناه قاعدة واحدة: اكتب دالة تأخذ المدخلات وتعيد المخرجات، دون أي آثار جانبية. ثم تحويله.

## The Four Transforms

### grad -- Differentiate a function

```python
grads = jax.grad(loss_fn)(params, x, y)
loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
```

استخدم عندما: تحتاج إلى تدرجات للتحسين.
القيد: يجب أن تقوم الدالة بإرجاع عددية. بالنسبة للمخرجات غير العددية، استخدم `jax.jacobian`.

### jit -- Compile a function

```python
fast_fn = jax.jit(f)
```

استخدم عندما: سيتم استدعاء الوظيفة أكثر من مرة باستخدام نفس المدخلات.
القيد: لا يوجد تدفق تحكم في Python يعتمد على القيم المتتبعة. استخدم `jax.lax.cond` للشروط الشرطية، `jax.lax.scan` للحلقات.

### vmap -- Vectorize a function

```python
batch_fn = jax.vmap(f, in_axes=(None, 0))
```

استخدم عندما: كتبت دالة لمثال واحد وتحتاج إليها للعمل على دفعات.
`in_axes` يحدد محور الوسيطة الذي سيتم الدفع عليه. `None` تعني عدم الدفع (البث).

### pmap -- Parallelize across devices

```python
parallel_fn = jax.pmap(f, axis_name='devices')
```

استخدم عندما: لديك عدة GPUs/TPUs وتريد توازي البيانات.
داخل الدالة، `jax.lax.pmean(x, 'devices')` المتوسطات عبر الأجهزة.

## Composition Rules

التحويلات تؤلف. الترتيب مهم:

```python
per_example_grads = jax.jit(jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0)))
```

القراءة من اليمين إلى اليسار: خذ تدرج الخسارة_fn، ثم قم بتوجيه الأمثلة، ثم قم بتجميع النتيجة.

التراكيب الصالحة:
- `jit(grad(f))` -- حساب التدرج المجمع
- `jit(vmap(f))` -- حسابات مجمعة مجمعة
- `vmap(grad(f))` -- التدرجات لكل مثال
- `pmap(jit(f))` -- حساب مجمع متوازي
- `grad(jit(f))` -- تدرج الدالة المترجمة (مثل jit(grad(f)))

## Parameter Management Pattern

JAX المعلمات عبارة عن pytrees (إملاءات متداخلة للمصفوفات):

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 10)),  'b': jnp.zeros(10)},
}
```

تحديث جميع المعلمات مرة واحدة:
```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

عدد المعلمات:
```python
n_params = sum(p.size for p in jax.tree.leaves(params))
```

## PRNG Key Management

JAX يتطلب مفاتيح عشوائية صريحة:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
noise = jax.random.normal(subkey, shape)
```

بالنسبة لعمليات عشوائية متعددة، قم بالتقسيم مرة واحدة:
```python
keys = jax.random.split(key, n)
```

لا تعيد استخدام المفتاح أبدًا. قم بتقسيمها دائمًا قبل الاستخدام.

## Common Mistakes

1. ** المصفوفات المتغيرة داخل jit **: المصفوفات JAX غير قابلة للتغيير. استخدم `x.at[i].set(v)` بدلاً من `x[i] = v`.

2. **استخدام طباعة Python داخل jit**: يتم تشغيل `print` أثناء التتبع، وليس التنفيذ. استخدم `jax.debug.print("{}", x)`.

3. **Python if/for inside jit على القيم المتتبعة**: استخدم `jax.lax.cond`، `jax.lax.switch`، `jax.lax.scan`، `jax.lax.fori_loop`.

4. **نسيان `.block_until_ready()`**: JAX يستخدم الإرسال غير المتزامن. لقياس الأداء، اتصل بـ `.block_until_ready()` لانتظار الاكتمال الفعلي.

5. **إعادة استخدام مفاتيح PRNG**: عمليتان بنفس المفتاح تنتج نفس القيم "العشوائية". انقسام دائما.

6. **الحالة العامة في الوظائف المتوترة**: يتم التقاط المتغيرات العامة في وقت التتبع. التغييرات بعد التتبع غير مرئية. تمرير كل شيء كحجج.

## Decision Checklist

1. هل يتم استدعاء الدالة أكثر من مرة؟ أضف `@jax.jit`.
2. هل يحتاج إلى تدرجات؟ التفاف مع `jax.grad` أو `jax.value_and_grad`.
3. هل تتم معالجة مثال واحد ولكن لديك دفعة؟ لف بـ `jax.vmap`.
4. هل لديك أجهزة متعددة؟ التفاف مع `jax.pmap`.
5. هل يستخدم العشوائية؟ خيط PRNG مفاتيح من خلال صراحة.
6. هل لديها تدفق تحكم بايثون على قيم المصفوفة؟ استبدل بـ `jax.lax` البدائيات.

## When to Use JAX

استخدم JAX عندما:
- أنت بحاجة إلى تدرجات لكل مثال (الخصوصية التفاضلية، معلومات فيشر)
- أنت تتدرب على وحدات TPU (JAX هو الإطار الأصلي)
- أنت بحاجة إلى مشتقات ذات ترتيب أعلى (الهسيين، اليعقوبيين)
- تريد تجميع خطوة التدريب بأكملها في نواة واحدة
- فريقك موجود في Google DeepMind أو Anthropic

استخدم PyTorch عندما:
- تريد أكبر نظام بيئي (HuggingFace، torchvision، Lightning)
- أنت تعطي الأولوية لسهولة تصحيح الأخطاء على السرعة الأولية
- أنت تقوم بالنشر إلى NVIDIA GPUs مع TorchServe/Triton
- أنت تقوم بالتوظيف (يوجد المزيد من المطورين PyTorch)
- تريد التكرار بسرعة على البنى الجديدة
