---
name: skill-latency-profiler
description: Write a complete latency-benchmarking script with warmup, synchronisation, percentiles, and memory tracking
version: 1.0.0
phase: 4
lesson: 15
tags: [edge, deployment, profiling, benchmarking]
---

# ملف التعريف الكمون
قم بإنشاء معيار زمن استجابة منضبط لأي نموذج PyTorch. التقارير التي يمكن لأي شخص أن يثق بها في الواقع.
##متى يستخدم
- مقارنة الأعمدة الأساسية المتعددة للمرشحين قبل اختيار واحدة للنشر.
- قبل وبعد التكميم أو التقليم.
- بعد تغيير وقت التشغيل (حريص مقابل ONNX مقابل TensorRT).
- إنشاء تقرير جاهزية النشر.
## المدخلات
- `model`: PyTorch `nn.Module`.
- `input_shape`: صف مثل `(1, 3, 224, 224)`.
- `device`: `cpu` | __الكود_6__ | __الكود_7__.
- `warmup`: الافتراضي 10.
- `iters`: الافتراضي 100.
## الشيكات
### 1. الاحماء
قم بتشغيل النموذج `warmup` مرات بدون توقيت. يلتقط تجميع JIT للأمام أولاً وتأثيرات ذاكرة التخزين المؤقت الباردة.
### 2. التزامن
بالنسبة إلى `cuda`، اتصل بـ `torch.cuda.synchronize()` قبل وبعد كل تمريرة للأمام المحددة بوقت.
بالنسبة لـ `mps`، اتصل بـ `torch.mps.synchronize()`.
### 3. الموقت
استخدم `time.perf_counter()` لقياس ساعة الحائط. تحويل إلى ميلي ثانية.
### 4. النسب المئوية
فرز القائمة الكاملة للتوقيتات. تقرير `p50, p90, p95, p99, mean, std`.
### 5. الذاكرة
بالنسبة لـ `cuda`، اتصل بـ `torch.cuda.max_memory_allocated()` بعد التشغيل واطرح أي خط أساس.
بالنسبة إلى `cpu`، استخدم `tracemalloc` أو `psutil.Process().memory_info().rss` قبل وبعد.
### 6. اكتساح حجم الدفعة
كرر بشكل اختياري المعيار الخاص بـ `batch_size in [1, 4, 16, 32]` للكشف عن المفاضلات بين الإنتاجية وزمن الاستجابة.
## قالب الإخراج
```python
import time
import torch
import psutil, os

def profile(model, input_shape, device="cpu", warmup=10, iters=100):
    proc = psutil.Process(os.getpid())
    baseline_rss = proc.memory_info().rss / 1e6

    model = model.to(device).eval()
    x = torch.randn(input_shape, device=device)

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        sync()
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        times = []
        for _ in range(iters):
            sync()
            t0 = time.perf_counter()
            model(x)
            sync()
            times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    mean = sum(times) / len(times)
    std  = (sum((t - mean) ** 2 for t in times) / len(times)) ** 0.5

    def pct(p):
        idx = max(0, min(len(times) - 1, int(len(times) * p) - 1))
        return times[idx]

    report = {
        "p50_ms":  pct(0.50),
        "p90_ms":  pct(0.90),
        "p95_ms":  pct(0.95),
        "p99_ms":  pct(0.99),
        "mean_ms": mean,
        "std_ms":  std,
        "rss_mb":  proc.memory_info().rss / 1e6 - baseline_rss,
    }
    if device == "cuda":
        report["peak_cuda_mb"] = torch.cuda.max_memory_allocated() / 1e6

    return report
```

## قواعد
- قم دائمًا بإجراء عملية الإحماء؛ لا تثق أبدًا في التوقيت الأول.
- النسب المئوية، ليست متوسطة - يمكن للقيمة المتطرفة الواحدة مضاعفة المتوسط ​​ولكنها بالكاد تتحرك p50.
- استخدم نفس input_shape للإنتاج؛ الكمون على 224x224 ليس الكمون على 384x384.
- بالنسبة إلى CUDA، لا تحذف أبدًا `torch.cuda.synchronize()`؛ الأرقام لا معنى لها بدونها.
- قم بتسجيل إصدار الشعلة، وإصدار CUDA، واسم الجهاز إلى جانب الأرقام - ولن تكون قابلة للمقارنة بخلاف ذلك.