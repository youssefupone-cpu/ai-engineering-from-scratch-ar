# نقطة تفتيش التدرج وإعادة حساب التنشيط
> يحافظ Backprop على كل عملية تنشيط وسيطة. عند 70 بايت من المعلمات وسياق 128 كيلو بايت، يكون ذلك 3 TB من عمليات التنشيط لكل رتبة. تعمل نقطة التفتيش على استبدال FLOPs بالذاكرة: إعادة الحساب بدلاً من الحفظ. السؤال هو ما هي الأجزاء التي يجب إسقاطها، والإجابة ليست "جميعها".
**النوع:** بناء
** اللغات: ** بايثون (مع شعلة اختيارية numpy)
**المتطلبات الأساسية:** المرحلة 10 الدرس 04 (التدريب المسبق المصغر-GPT)، المرحلة 10 الدرس 05 (القياس والتوزيع)
**الوقت:** ~70 دقيقة
## المشكلة
تدريب المحول يخزن، لكل طبقة، المدخلات لكل عملية يتم تمييزها في الخلف: مدخلات الانتباه، وإسقاطات Q/K/V، ومخرجات softmax، ومدخلات FFN، والمخرجات المعيارية، والتيار المتبقي. بالنسبة للطبقة ذات الحجم المخفي `d`، طول التسلسل `L`، الدفعة `B`، يكون هذا بترتيب `12 * B * L * d` العوامات لكل طبقة.
بالنسبة إلى `d=8192, L=8192, B=1`، هذا يعني 800 MB/طبقة في BF16. النموذج المكون من 64 طبقة هو 51 GB من عمليات التنشيط - وذلك قبل الضرب بحجم الدفعة الصغيرة، وقبل إضافة وسيطات انتباه softmax (`L^2` لكل رأس)، وقبل تحليل النسخ الجزئية المتوازية.
الفاتورة ذات الوجهين: BF16 قد تتناسب الأوزان بالإضافة إلى حالة المحسن مع 80 جيجابايت، لكن عمليات التنشيط تدفعك إلى تجاوزها. تعتبر نقاط التفتيش المتدرجة (المعروفة أيضًا باسم إعادة حساب التنشيط) هي الحل القياسي. إسقاط معظم عمليات التنشيط؛ إعادة الأمام أثناء الخلف لاستعادتهم. التكلفة: تقلبات إضافية. الفائدة: تنخفض الذاكرة بنسبة أجزاء نقطة التفتيش إلى إجمالي الطبقات.
إذا تم القيام بذلك بسذاجة، فإن نقاط التفتيش تكلف ما يقرب من 33٪ من عمليات التمرير الأمامية في كل خطوة. أحسنت صنعًا - نقاط تفتيش انتقائية وفقًا "للاختيار الذكي" لكورثيكانتي وآخرين. — يمكنك توفير ذاكرة بمقدار 5x لأقل من 5% FLOP من الحمل الزائد. ومع FP8 matmuls، وFSDP تفريغ، وMoE الموازي للخبراء، فإن هذا مهم حقًا: لا يمكنك تحمل تكاليف الذاكرة أو الحوسبة المهدرة.
##المفهوم
### ما يحتاجه Backward فعليًا
__الكود_0__. الرجوع للخلف يريد `grad_input` و`grad_params`. لحسابها يحتاج إلى:
- `input` (لحساب `grad_params = input.T @ grad_output` للطبقات الخطية)
- بعض وسيطات مشتقات التنشيط (يعتمد مشتق ReLU/GELU/softmax على قيمة التنشيط)
يقوم التمرير الأمامي بتخزين هذه العناصر تلقائيًا في الرسم البياني التلقائي. كل `tensor.retain_grad()` وكل عملية تحتاج إلى مدخلاتها تحتفظ بمرجع.
### نقاط التفتيش الكاملة الساذجة
قم بتقسيم الشبكة إلى أجزاء `N`. أثناء التقدم، قم بتخزين *الإدخال* فقط لكل قطعة. عندما تكون هناك حاجة للخلف في الوسط، أعد تشغيل التمريرة الأمامية للمقطع لتجسيدها، ثم قم بالتفريق بينها.
مثال: محول مكون من 32 طبقة مقسم إلى 32 قطعة مكونة من طبقة واحدة لكل منها.
- الذاكرة: 32 طبقة مدخلة (صغيرة) مقابل 32 * (حجم التنشيط لكل طبقة) (ضخمة).
- حساب إضافي: 1 إضافي للأمام لكل مقطع، أي إجمالي 33% من عمليات التقليب للأمام (بما أن الخلف هو 2x للأمام، تصبح الخطوة الكاملة 1 + 1 + 2 = 4 وحدات بدلاً من 1 + 2 = 3).
هذا هو تشن الأصلي وآخرون. وصفة 2016: نقطة تفتيش واحدة في كل `sqrt(L)` من الطبقات لتحقيق التوازن بين الذاكرة والحوسبة. بالنسبة لـ L=64، هذا يعني 8 نقاط تفتيش.
### نقاط التفتيش الانتقائية (كورثيكانتي 2022)
ليست كل عمليات التنشيط بنفس التكلفة. إخراج انتباه softmax هو `B*L*L*heads` وينمو *بشكل تربيعي* مع طول التسلسل. التنشيط المخفي FFN هو `B*L*4d` وينمو بشكل خطي. بالنسبة للتسلسلات الطويلة، يهيمن softmax.
تحافظ نقاط التفتيش الانتقائية على عمليات التنشيط الرخيصة للتخزين (التوقعات الخطية والمتبقية) وتعيد حساب التنشيطات الباهظة الثمن فقط (الانتباه). أنت تدفع الحد الأدنى من FLOPs لإعادة الحساب مع حفظ ذاكرة O(L^2).
تقوم Megatron-Core بتنفيذ هذا كإعادة حساب التنشيط "الانتقائي". يُستخدم في معظم جولات التدريب على الحدود التي يزيد عددها عن 2024+.
### تفريغ
بديل لإعادة الحساب: قم بإرسال عمليات التنشيط إلى CPU RAM بين الأمام والخلف. يتطلب عرض النطاق الترددي PCIe؛ يكون مفيدًا عندما يتجاوز عرض النطاق الترددي الخامل تكلفة إعادة التجسيد. الاستراتيجيات المختلطة شائعة: قم بفحص بعض الطبقات وتفريغ طبقات أخرى.
FSDP2 يتم تفريغ السفن كخيار من الدرجة الأولى. يظهر إلغاء التحميل عندما يكون GPU في حالة اختناق في الذاكرة ولكن نقل CPU-GPU له مساحة رأسية.
### نموذج إعادة حساب التكلفة
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

### نموذج توفير الذاكرة
حجم التنشيط لكل طبقة: `A`. بالنسبة لطبقات `L`، إجمالي ذاكرة التنشيط: `L * A`.
نقطة تفتيش كاملة (حجم المقطع 1): قم بتخزين `L * input_volume` فقط (~`L * 1/10 A` للمحول القياسي). يحفظ ~`9 * L * A * 1/10`.
نقطة تفتيش كل `k` طبقات: قم بتخزين `L/k * A` بالإضافة إلى `k-1` قيمة الطبقات داخل المقطع النشط.
عند `k = sqrt(L)`، يتم قياس تكلفة الذاكرة وإعادة الحساب باستخدام `sqrt(L)` - وهي المفاضلة المثالية لطبقات التكلفة الموحدة.
### متى لا تذهب إلى نقطة التفتيش
- الطبقات الأعمق لمرحلة pipeline أثناء الطيران بالفعل. عليهم أن ينتهوا على أي حال.
- الطبقتان الأولى والأخيرة إذا سيطرتا على حساب المرحلة (نادرًا في المحولات).
- نواة الانتباه تستخدم FlashAttention بالفعل — يقوم Flash بالفعل بإعادة حساب softmax بسرعة، لذا فإن نقاط التفتيش الإضافية على مستوى الطبقة تضيف القليل في الأعلى.
### أنماط التنفيذ
1. **مجمّع الدالة:** قم بتغليف مقطع في `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch يخزن `input` فقط، ويعيد حساب كل شيء آخر في الخلف.
2. **المعتمد على الديكور:** قم بتسمية الطبقات باعتبارها قابلة للفحص؛ يقرر المدرب في وقت التكوين الأجزاء التي سيتم تغليفها.
3. **إعادة الحساب الصريح يدويًا:** اكتب التمريرة الخلفية بنفسك، واستدعاء `recompute_forward` المخصص الذي يكرر الإرسال الأمامي مع الإدخال المخزن.
الثلاثة جميعهم يعطيون نفس النتيجة الوظيفية. الأغلفة هي المصطلح القياسي.
### التفاعل مع TP / PP / FP8
- **الموتر المتوازي:** يجب جمع مدخلات نقطة التفتيش أو إعادة تشتيتها عند إعادة الحساب؛ التعامل مع تكلفة الاتصالات.
- **خط الأنابيب المتوازي:** النمط النموذجي هو وضع نقطة تفتيش في كل pipeline-stage للأمام حتى تتمكن الدفعات الصغيرة ذات الترتيب العكسي من إعادة استخدام ذاكرة التنشيط.
- **FP8 إعادة الحساب:** يجب أن تتطابق تواريخ الحد الأقصى التي تم تحديثها أثناء إعادة الحساب مع التقدم الأصلي، أو انجرافات مقياس FP8. معظم الأطر تلتقط المقياس.
## بنائها
### الخطوة 1: نموذج لعبة يحتوي على أجزاء
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

### الخطوة الثانية: التراجع الساذج الذي يحتاج إلى كافة عمليات التنشيط
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

### الخطوة 3: نقطة تفتيش - كل ذاكرة
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

### الخطوة 4: نموذج التكلفة
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

### الخطوة 5: مقدر الذاكرة
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

### الخطوة 6: الحجم الأمثل للقطعة
```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### الخطوة 7: قرار نقطة التفتيش الانتقائية
```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## استخدمه
- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — المُجمِّع الأساسي في PyTorch. يلتف وظيفة؛ يخزن المدخلات فقط، ويعيد الحساب إلى الخلف.
- **إعادة حساب تنشيط Megatron-Core**: يدعم الأوضاع `selective`، `full`، و`block`. قياسي في 2024+ التدريب الحدودي.
- **FSDP2 إلغاء التحميل**: `module.to_empty(device="cpu")` مع `offload_policy` في عمليات تنشيط الأجزاء FSDP2 إلى CPU بدلاً من إعادة الحساب.
- **DeepSpeed ​​ZeRO-Offload**: CPU إلغاء التحميل لحالات المُحسِّن وعمليات التنشيط، واستكمال عملية التحقق.
## اشحنها
يُنتج هذا الدرس `outputs/prompt-activation-recompute-policy.md` — مطالبة تأخذ تكوين النموذج الخاص بك (الطبقات، المخفية، التسلسلية، الدفعية) وذاكرة GPU المتاحة وتصدر سياسة إعادة حساب لكل طبقة (لا شيء / انتقائي / كامل / إلغاء التحميل).
## تمارين
1. التحقق من الصحة. قم بتشغيل `model_forward` + `model_backward` (عمليات التنشيط الكاملة) مقابل `model_forward_checkpointed` + `model_backward_checkpointed` (المقاطع). يجب أن تكون تدرجات المعلمات مطابقة لدقة الماكينة.
2. قم بمسح حجم المقطع `k` من 1 إلى `L`. قطعة الأرض FLOP الحمل والذاكرة. أوجد ركبة المنحنى.
3. تنفيذ نقاط التفتيش الانتقائية: قم بتخزين مدخلات وحدة الانتباه وليس الوسائط الوسيطة. قم بقياس الحمل الزائد FLOP مقابل نقاط فحص الطبقة الكاملة لنموذج مكون من 32 طبقة بالتسلسل = 8192.
4. إضافة التفريغ. حفظ مدخلات القطعة إلى "CPU المخزن المؤقت" المحاكاة (قائمة منفصلة). قم بقياس "عرض النطاق الترددي PCIe" بالبايت/الوقت وابحث عن نقطة التعادل بين التفريغ وإعادة الحساب.
5. قم بقياس محول PyTorch حقيقي مع وبدون `torch.utils.checkpoint`. قياس الذاكرة (عبر `torch.cuda.max_memory_allocated`) ووقت الخطوة.
## المصطلحات الرئيسية
| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| نقاط تفتيش التدرج | "حفظ الذاكرة عن طريق إعادة الأمام" | مدخلات قطاع المتجر فقط؛ أعد حساب الوسطيات أثناء الرجوع للخلف للحصول على موترات دعم التدرج |
| إعادة حساب التنشيط | "مثل نقاط التفتيش" | الاسم ذو النكهة HPC لنفس التقنية |
| حجم القطعة (ك) | "كم عدد الطبقات لكل نقطة تفتيش" | عدد الطبقات التي تم إسقاط وسيطاتها وإعادة تجسيدها معًا |
| نقاط التفتيش الانتقائية | "خدعة كورثيكانتي" | إعادة حساب عمليات التنشيط المكلفة فقط للمتجر (انتباه softmax)؛ احتفظ بالأشياء الرخيصة |
| فحص كامل | "النسخة الساذجة" | أعد حساب الوسطيات لكل طبقة في كل قطعة |
| كتلة نقاط التفتيش | "خشن الحبيبات" | نقطة تفتيش كتل المحولات بأكملها؛ أكبر تفصيل |
| FLOP النفقات العامة | "ضريبة الحساب" | عمليات التقليب الإضافية لكل خطوة = (إعادة حساب عمليات التقليب) / (fwd + bwd FLOPs)؛ 33% ساذج، 5% انتقائي |
| تفريغ التنشيط | "الشحن إلى CPU" | انقل عمليات التنشيط إلى CPU RAM عبر الأمام->الخلف؛ بديل لإعادة الحساب |
| قاعدة sqrt-L | "الأمثل الكلاسيكي" | بالنسبة للطبقات ذات التكلفة الموحدة، فإن التباعد الأمثل لنقاط التفتيش هو طبقات sqrt(L) |
| الاهتمام-softmax حجم | "مشكلة O(L^2)" | L ^ 2 * رؤوس * عوامات الدفعة؛ يهيمن على ذاكرة التنشيط في السياقات الطويلة |
## مزيد من القراءة
- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- الورقة الأصلية التي أضفت طابعًا رسميًا على فحص التدرج اللوني
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- إعادة حساب التنشيط الانتقائي والتحليل الرسمي للتكلفة
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- نهج بديل للذاكرة الثابتة من خلال إعادة تجسيد الوضع العكسي
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- إلغاء تحميل التنشيط على نطاق واسع
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- المعيار API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) - الأوضاع الانتقائية والكاملة والحظرية