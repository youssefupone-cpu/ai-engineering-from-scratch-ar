# GPU Setup & Cloud

> التدريب على CPU جيد للتعلم. التدريب على الاحتياجات الحقيقية أ GPU.

**النوع:** بناء
** اللغات: ** بايثون
**المتطلبات الأساسية:** المرحلة 0، الدرس 01
**الوقت:** ~45 دقيقة

## Learning Objectives

- تحقق من توفر GPU المحلي باستخدام `nvidia-smi` وPyTorch's CUDA API
- قم بتكوين Google Colab بـ T4 GPU للتجارب السحابية المجانية
- ضرب المصفوفة المعيارية على CPU مقابل GPU وقياس السرعة
- قم بتقدير أكبر نموذج يناسب VRAM الخاص بك باستخدام قاعدة fp16 الأساسية

## The Problem

معظم الدروس في المراحل 1-3 تعمل بشكل جيد على CPU. ولكن بمجرد أن تبدأ في تدريب شبكات CNN أو المحولات أو LLMs (المراحل 4+)، فإنك تحتاج إلى تسارع GPU. الجري التدريبي الذي يستغرق 8 ساعات على CPU يستغرق 10 دقائق على GPU.

لديك ثلاثة خيارات: محلي GPU، سحابي GPU، أو Google Colab (مجاني).

## The Concept

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

## Build It

### Option 1: Local NVIDIA GPU

تحقق مما إذا كان لديك واحدة:

```bash
nvidia-smi
```

تثبيت PyTorch مع CUDA:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### Option 2: Google Colab

1. انتقل إلى [colab.research.google.com](https://colab.research.google.com)
2. وقت التشغيل > تغيير نوع وقت التشغيل > T4 GPU
3. قم بتشغيل `!nvidia-smi` للتحقق

قم بتحميل دفاتر الملاحظات من هذه الدورة التدريبية مباشرةً إلى Colab.

### Option 3: Cloud GPU

بالنسبة إلى Lambda Labs أو RunPod أو Vast.ai:

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### No GPU? No problem.

تعمل معظم الدروس على CPU. الأشخاص الذين يحتاجون إلى GPU سيقولون ذلك ويتضمنون روابط Colab.

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## Build It: GPU vs CPU benchmark

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

## Exercises

1. قم بتشغيل المعيار أعلاه وقارن CPU مقابل GPU مرة
2. إذا لم يكن لديك GPU، قم بتشغيله على Google Colab وقارن
3. تحقق من مقدار الذاكرة GPU المتوفرة لديك وقم بتقدير أكبر نموذج يمكنك ملاءمته (القاعدة الأساسية: 2 بايت لكل معلمة لـ fp16)

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
| CUDA | "GPU البرمجة" | منصة الحوسبة المتوازية NVIDIA التي تتيح لك تشغيل التعليمات البرمجية على GPU |
| VRAM | "GPU الذاكرة" | فيديو RAM على GPU منفصل عن النظام RAM. يحد من حجم النموذج. |
| FP16 | "نصف الدقة" | الفاصلة العائمة 16 بت، تستخدم نصف ذاكرة fp32 مع الحد الأدنى من فقدان الدقة |
| الموتر الأساسية | "أجهزة المصفوفة السريعة" | نوى GPU متخصصة لمضاعفة المصفوفات، أسرع بـ 4-8 مرات من النوى العادية |
