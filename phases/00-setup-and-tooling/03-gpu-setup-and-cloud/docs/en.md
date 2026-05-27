# إعداد GPU والسحابة

> التدريب على وحدة المعالجة المركزية أمر جيد للتعلم. التدريب الحقيقي يحتاج إلى GPU.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 0، الدرس 01
**الوقت:** ~45 دقيقة

## أهداف التعلم

- تحقق من توفر وحدة معالجة الرسومات المحلية باستخدام `nvidia-smi` وواجهة برمجة تطبيقات CUDA الخاصة بـ PyTorch
- قم بتكوين Google Colab باستخدام وحدة معالجة الرسومات T4 لإجراء تجارب سحابية مجانية
- ضرب المصفوفة المعيارية على وحدة المعالجة المركزية مقابل وحدة معالجة الرسومات وقياس السرعة
- قم بتقدير الطراز الأكبر الذي يتناسب مع ذاكرة الفيديو (VRAM) الخاصة بك باستخدام القاعدة الأساسية fp16

## المشكلة

تعمل معظم الدروس في المراحل 1-3 بشكل جيد على وحدة المعالجة المركزية. ولكن بمجرد البدء في تدريب شبكات CNN أو المحولات أو LLMs (المراحل 4+)، ستحتاج إلى تسريع GPU. يستغرق التدريب الذي يستغرق 8 ساعات على وحدة المعالجة المركزية 10 دقائق على وحدة معالجة الرسومات.

لديك ثلاثة خيارات: وحدة معالجة الرسومات المحلية، أو وحدة معالجة الرسومات السحابية، أو Google Colab (مجانًا).

##المفهوم

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

## بنائها

### الخيار 1: وحدة معالجة الرسومات NVIDIA المحلية

تحقق مما إذا كان لديك واحدة:

```bash
nvidia-smi
```

تثبيت PyTorch باستخدام CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### الخيار 2: جوجل كولاب

1. انتقل إلى [colab.research.google.com](https://colab.research.google.com)
2. وقت التشغيل > تغيير نوع وقت التشغيل > T4 GPU
3. قم بتشغيل `!nvidia-smi` للتحقق

قم بتحميل دفاتر الملاحظات من هذه الدورة التدريبية مباشرةً إلى Colab.

### الخيار 3: وحدة معالجة الرسومات السحابية

بالنسبة إلى Lambda Labs أو RunPod أو Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### لا يوجد GPU؟ لا مشكلة.

تعمل معظم الدروس على وحدة المعالجة المركزية (CPU). الأشخاص الذين يحتاجون إلى GPU سيقولون ذلك ويتضمنون روابط Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## بنائها: معيار GPU مقابل وحدة المعالجة المركزية

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## تمارين

1. قم بتشغيل المعيار أعلاه وقارن بين أوقات وحدة المعالجة المركزية ووحدة معالجة الرسومات
2. إذا لم يكن لديك وحدة معالجة رسومات، فقم بتشغيلها على Google Colab وقارنها
3. تحقق من مقدار ذاكرة وحدة معالجة الرسومات لديك وقم بتقدير أكبر نموذج يمكنك ملاءمته (القاعدة الأساسية: 2 بايت لكل معلمة لـ fp16)

## المصطلحات الرئيسية

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| كودا | "برمجة GPU" | منصة الحوسبة المتوازية من NVIDIA والتي تتيح لك تشغيل التعليمات البرمجية على وحدة معالجة الرسومات |
| VRAM | "ذاكرة وحدة معالجة الرسومات" | ذاكرة الوصول العشوائي للفيديو على وحدة معالجة الرسومات، منفصلة عن ذاكرة الوصول العشوائي للنظام. يحد من حجم النموذج. |
| FP16 | "نصف الدقة" | الفاصلة العائمة 16 بت، تستخدم نصف ذاكرة fp32 مع الحد الأدنى من فقدان الدقة |
| الموتر الأساسية | "أجهزة المصفوفة السريعة" | نوى وحدة معالجة الرسومات (GPU) المتخصصة لمضاعفة المصفوفات، أسرع بمعدل 4-8 مرات من النوى العادية |