# Gradient Checkpointing and Activation Recomputation

> يحافظ Backprop على كل عملية تنشيط وسيطة. عند 70 بايت من المعلمات وسياق 128 كيلو بايت يكون ذلك 3 TB من التنشيط لكل رتبة. تعمل نقطة التفتيش على استبدال FLOPs بالذاكرة: إعادة الحساب بدلاً من الحفظ. السؤال هو ما هي الأجزاء التي يجب إسقاطها، والإجابة ليست "جميعها".

**النوع:** بناء
** اللغات: ** بايثون (مع شعلة اختيارية numpy)
**المتطلبات الأساسية:** المرحلة 10 الدرس 04 (التدريب المسبق المصغر GPT)، المرحلة 10 الدرس 05 (القياس والتوزيع)
**الوقت:** ~70 دقيقة

## The Problem

تدريب المحول يخزن، لكل طبقة، المدخلات لكل عملية يتم تمييزها في الخلف: مدخلات الانتباه، وإسقاطات Q/K/V، ومخرجات softmax، والمدخلات FFN، والمخرجات المعيارية، والتيار المتبقي. بالنسبة للطبقة ذات الحجم المخفي `d`، طول التسلسل `L`، الدفعة `B`، يكون ذلك بترتيب `12 * B * L * d` العوامات لكل طبقة.

بالنسبة إلى `d=8192, L=8192, B=1`، يكون ذلك 800 MB/طبقة في BF16. النموذج المكون من 64 طبقة هو 51 GB من التنشيطات — وذلك قبل أن تتضاعف في حجم الدفعة الصغيرة، وقبل أن تضيف وسيطات انتباه softmax (`L^2` لكل رأس)، وقبل أن تقوم بتحليل النسخ الجزئية المتوازية.

الفاتورة ذات الوجهين: BF16 الأوزان بالإضافة إلى حالة المحسن قد تناسب 80 جيجابايت، لكن عمليات التنشيط تدفعك إلى تجاوزها. تعتبر نقاط التفتيش المتدرجة (المعروفة أيضًا باسم إعادة حساب التنشيط) هي الحل القياسي. إسقاط معظم عمليات التنشيط؛ إعادة الأمام أثناء الخلف لاستعادتهم. التكلفة: تقلبات إضافية. الفائدة: تنخفض الذاكرة بنسبة أجزاء نقطة التفتيش إلى إجمالي الطبقات.

إذا تم القيام بذلك بسذاجة، فإن نقاط التفتيش تكلف ما يقرب من 33٪ من عمليات التمرير الأمامية في كل خطوة. أحسنت صنعًا - نقاط تفتيش انتقائية وفقًا "للاختيار الذكي" لكورثيكانتي وآخرين. - يمكنك توفير ذاكرة 5x بأقل من 5% FLOP من النفقات العامة. ومع FP8 matmuls، FSDP إلغاء التحميل، وMoE الموازي للخبراء، هذا مهم حقًا: لا يمكنك تحمل تكلفة الذاكرة أو الحوسبة الضائعة.

## The Concept

### What Backward Actually Needs

`output = layer(input)`. يريد الخلف `grad_input` و `grad_params`. لحسابها يحتاج إلى:

- `input` (لحساب `grad_params = input.T @ grad_output` للطبقات الخطية)
- بعض وسيطات مشتقات التنشيط (مشتق ReLU/GELU/softmax يعتمد على قيمة التنشيط)

يقوم التمرير الأمامي بتخزين هذه العناصر تلقائيًا في الرسم البياني التلقائي. كل `tensor.retain_grad()` وكل عملية تحتاج إلى مدخلاتها تحتفظ بمرجع.

### Naive Full Checkpointing

قم بتقسيم الشبكة إلى مقاطع `N`. أثناء التقدم، قم بتخزين *الإدخال* فقط لكل قطعة. عندما تكون هناك حاجة للخلف في الوسط، أعد تشغيل التمريرة الأمامية للمقطع لتجسيدها، ثم قم بالتفريق بينها.

مثال: محول مكون من 32 طبقة مقسم إلى 32 قطعة مكونة من طبقة واحدة لكل منها.

- الذاكرة: 32 طبقة مدخلة (صغيرة) مقابل 32 * (حجم التنشيط لكل طبقة) (ضخمة).
- حساب إضافي: 1 إضافي للأمام لكل مقطع، أي إجمالي 33% من عمليات التقليب للأمام (بما أن الخلف هو 2x للأمام، تصبح الخطوة الكاملة 1 + 1 + 2 = 4 وحدات بدلاً من 1 + 2 = 3).

هذا هو تشن الأصلي وآخرون. وصفة 2016: نقطة تفتيش واحدة كل `sqrt(L)` من الطبقات لتحقيق التوازن بين الذاكرة والحوسبة. بالنسبة لـ L=64، هذا يعني 8 نقاط تفتيش.

### Selective Checkpointing (Korthikanti 2022)

ليست كل عمليات التنشيط بنفس التكلفة. إخراج انتباه softmax هو `B*L*L*heads` وينمو *بشكل تربيعي* مع طول التسلسل. التنشيط المخفي FFN هو `B*L*4d` وينمو خطيًا. بالنسبة للتسلسلات الطويلة، يهيمن softmax.

تحافظ نقاط التفتيش الانتقائية على عمليات التنشيط الرخيصة للتخزين (التوقعات الخطية والمتبقية) وتعيد حساب التنشيطات الباهظة الثمن فقط (الانتباه). أنت تدفع الحد الأدنى من FLOPs لإعادة الحساب مع حفظ ذاكرة O(L^2).

تقوم Megatron-Core بتنفيذ هذا كإعادة حساب التنشيط "الانتقائي". يُستخدم في معظم جولات التدريب على الحدود التي يزيد عددها عن 2024+.

### Offload

بديل لإعادة الحساب: قم بإرسال التنشيطات إلى CPU RAM بين الأمام والخلف. يتطلب عرض النطاق الترددي PCIe؛ يكون مفيدًا عندما يتجاوز عرض النطاق الترددي الخامل تكلفة إعادة التجسيد. الاستراتيجيات المختلطة شائعة: قم بفحص بعض الطبقات وتفريغ طبقات أخرى.

FSDP2 تفريغ السفن كخيار من الدرجة الأولى. يضيء التحميل عندما يكون GPU عنق الزجاجة في الذاكرة ولكن النقل CPU-GPU له مساحة رأسية.

### Recompute Cost Model

FLOPs لكل خطوة مع نقاط تفتيش ساذجة لكل `k` طبقات من `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

باستخدام نقاط التفتيش الانتقائية، يمكنك إعادة حساب نواة الانتباه فقط، وليس الطبقة بأكملها:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Memory Savings Model

حجم التنشيط لكل طبقة: `A`. بالنسبة لـ `L` الطبقات، إجمالي ذاكرة التنشيط: `L * A`.

نقطة تفتيش كاملة (حجم المقطع 1): قم بتخزين فقط `L * input_volume` (~`L * 1/10 A` للمحول القياسي). يحفظ ~`9 * L * A * 1/10`.

قم بوضع علامة تحقق عند كل `k` طبقات: قم بتخزين قيمة `L/k * A` زائد `k-1` الطبقات داخل الجزء النشط.

عند `k = sqrt(L)`، يتم قياس تكلفة الذاكرة وإعادة الحساب مع `sqrt(L)` — وهي المفاضلة المثالية لطبقات التكلفة الموحدة.

### When Not to Checkpoint

- الطبقات الأعمق لمرحلة pipeline أثناء الطيران بالفعل. عليهم أن ينتهوا على أي حال.
- الطبقتان الأولى والأخيرة إذا سيطرتا على حساب المرحلة (نادرًا في المحولات).
- نواة الانتباه تستخدم FlashAttention بالفعل — يقوم Flash بالفعل بإعادة حساب softmax بسرعة، لذا فإن نقاط التفتيش الإضافية على مستوى الطبقة تضيف القليل في الأعلى.

### Implementation Patterns

1. **مجمّع الدالة:** قم بتغليف مقطع في `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch يخزن فقط `input`، ويعيد حساب كل شيء آخر في الخلف.

2. **المعتمد على الديكور:** قم بتسمية الطبقات باعتبارها قابلة للفحص؛ يقرر المدرب في وقت التكوين الأجزاء التي سيتم تغليفها.

3. **إعادة الحساب الصريح يدويًا:** اكتب التمريرة الخلفية بنفسك، واستدعاء `recompute_forward` مخصص يكرر التمرير الأمامي مع الإدخال المخزن.

الثلاثة جميعهم يعطيون نفس النتيجة الوظيفية. الأغلفة هي المصطلح القياسي.

### Interaction with TP / PP / FP8

- **الموتر المتوازي:** يجب جمع مدخلات نقطة التفتيش أو إعادة تشتيتها عند إعادة الحساب؛ التعامل مع تكلفة الاتصالات.
- **خط الأنابيب الموازي:** النمط النموذجي هو وضع نقطة تفتيش في كل مرحلة pipخط للأمام حتى تتمكن الدفعات الصغيرة ذات الترتيب العكسي من إعادة استخدام ذاكرة التنشيط.
- **FP8 إعادة الحساب:** يجب أن تتطابق تواريخ amax التي تم تحديثها أثناء إعادة الحساب مع المهاجم الأصلي، أو انجرافات المقياس FP8. معظم الأطر تلتقط المقياس.

## Build It

### Step 1: A Toy Model With Segments

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Step 2: Naive Backward Needing All Activations

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Step 3: Checkpoint-Every-k Memory

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Step 4: Cost Model

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Step 5: Memory Estimator

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Step 6: Optimal Segment Size

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Step 7: Selective Checkpoint Decision

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Use It

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — الغلاف المتعارف عليه في PyTorch. يلتف وظيفة؛ يخزن المدخلات فقط، ويعيد الحساب إلى الخلف.
- **إعادة حساب تنشيط Megatron-Core**: يدعم الأوضاع `selective` و`full` و`block`. قياسي في 2024+ التدريب الحدودي.
- **FSDP2 إلغاء التحميل**: `module.to_empty(device="cpu")` مع `offload_policy` في عمليات تنشيط الأجزاء FSDP2 إلى CPU بدلاً من إعادة الحساب.
- **DeepSpeed ​​ZeRO-Offload**: CPU إلغاء التحميل لحالات المُحسّن وعمليات التنشيط، واستكمال عمليات التحقق.

## Ship It

ينتج هذا الدرس `outputs/prompt-activation-recompute-policy.md` — مطالبة تأخذ تكوين النموذج الخاص بك (الطبقات، المخفية، التسلسلية، الدفعية) والذاكرة المتوفرة GPU وتصدر سياسة إعادة حساب لكل طبقة (لا شيء / انتقائي / كامل / إلغاء التحميل).

## Exercises

1. التحقق من الصحة. تشغيل `model_forward` + `model_backward` (عمليات التنشيط الكاملة) مقابل `model_forward_checkpointed` + `model_backward_checkpointed` (المقاطع). يجب أن تكون تدرجات المعلمات مطابقة لدقة الماكينة.

2. قم بمسح حجم المقطع `k` من 1 إلى `L`. قطعة FLOP النفقات العامة والذاكرة. أوجد ركبة المنحنى.

3. تنفيذ نقاط التفتيش الانتقائية: قم بتخزين مدخلات وحدة الانتباه وليس الوسائط الوسيطة. قم بقياس FLOP الحمل الزائد مقابل نقاط التفتيش ذات الطبقة الكاملة لنموذج مكون من 32 طبقة بالتسلسل = 8192.

4. إضافة التفريغ. حفظ مدخلات المقطع في محاكاة "CPU المخزن المؤقت" (قائمة منفصلة). قم بقياس "عرض النطاق الترددي PCIe" بالبايت/الوقت وابحث عن نقطة التعادل بين التفريغ وإعادة الحساب.

5. قم بوضع علامة مرجعية على محول PyTorch حقيقي مع وبدون `torch.utils.checkpoint`. قياس الذاكرة (عبر `torch.cuda.max_memory_allocated`) ووقت الخطوة.

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| نقاط تفتيش التدرج | "حفظ الذاكرة عن طريق إعادة الأمام" | مدخلات قطاع المتجر فقط؛ أعد حساب الوسطيات أثناء الرجوع للخلف للحصول على موترات دعم التدرج |
| إعادة حساب التنشيط | "مثل نقاط التفتيش" | الاسم HPC ذو النكهة لنفس التقنية |
| حجم القطعة (ك) | "كم عدد الطبقات لكل نقطة تفتيش" | عدد الطبقات التي تم إسقاط وسيطاتها وإعادة تجسيدها معًا |
| نقاط التفتيش الانتقائية | "خدعة كورثيكانتي" | إعادة حساب عمليات التنشيط المكلفة فقط للمتجر (انتباه softmax)؛ احتفظ بالأشياء الرخيصة |
| فحص كامل | "النسخة الساذجة" | أعد حساب الوسطيات لكل طبقة في كل قطعة |
| كتلة نقاط التفتيش | "خشن الحبيبات" | نقطة تفتيش كتل المحولات بأكملها؛ أكبر تفصيل |
| FLOP النفقات العامة | "ضريبة الحساب" | عمليات التقليب الإضافية لكل خطوة = (إعادة حساب عمليات التقليب) / (fwd + bwd FLOPs)؛ 33% ساذج، 5% انتقائي |
| تفريغ التنشيط | "الشحن إلى CPU" | انقل عمليات التنشيط إلى CPU RAM عبر الأمام->الخلف؛ بديل لإعادة الحساب |
| قاعدة sqrt-L | "الأمثل الكلاسيكي" | بالنسبة للطبقات ذات التكلفة الموحدة، فإن التباعد الأمثل لنقاط التفتيش هو طبقات sqrt(L) |
| الاهتمام-softmax حجم | "مشكلة O(L^2)" | L ^ 2 * رؤوس * عوامات الدفعة؛ يهيمن على ذاكرة التنشيط في السياقات الطويلة |

## Further Reading

- [تشن وآخرون، 2016 - "تدريب الشبكات العميقة بتكلفة الذاكرة الفرعية"](https://arxiv.org/abs/1604.06174) - الورقة الأصلية التي أضفت طابعًا رسميًا على فحص التدرج
- [كورثيكانتي وآخرون، 2022 - "تقليل إعادة حساب التنشيط في نماذج المحولات الكبيرة"](https://arxiv.org/abs/2205.05198) - إعادة حساب التنشيط الانتقائي وتحليل التكلفة الرسمي
- [Pudipeddi وآخرون، 2020 -- "تدريب الشبكات العصبية الكبيرة باستخدام الذاكرة الثابتة باستخدام خوارزمية تنفيذ جديدة"](https://arxiv.org/abs/2002.05645) -- نهج بديل للذاكرة الثابتة عبر إعادة تجسيد الوضع العكسي
- [رين وآخرون، 2021 -- "ZeRO-Offload: إضفاء الطابع الديمقراطي على التدريب النموذجي على نطاق ملياري"](https://arxiv.org/abs/2101.06840) -- تفريغ التنشيط على نطاق واسع
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- المعيار API
- [وثائق إعادة حساب تنشيط Megatron-Core](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) - أوضاع انتقائية وكاملة وكتلة
