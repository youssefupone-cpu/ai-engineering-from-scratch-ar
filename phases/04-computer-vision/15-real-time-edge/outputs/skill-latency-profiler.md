---
name: skill-latency-profiler
description: Write a complete latency-benchmarking script with warmup, synchronisation, percentiles, and memory tracking
version: 1.0.0
phase: 4
lesson: 15
tags: [edge, deployment, profiling, benchmarking]
---

# Latency Profiler

قم بإنتاج معيار زمن انتقال منضبط لأي طراز PyTorch. التقارير التي يمكن لأي شخص أن يثق بها في الواقع.

## When to use

- مقارنة الأعمدة الأساسية المتعددة للمرشحين قبل اختيار واحدة للنشر.
- قبل وبعد التكميم أو التقليم.
- بعد تغيير وقت التشغيل (حريص مقابل ONNX مقابل TensorRT).
- إنشاء تقرير جاهزية النشر.

## Inputs

- `model`: PyTorch `nn.Module`.
- `input_shape`: صف مثل `(1, 3, 224, 224)`.
- `device`: `cpu` | `cuda` | `mps`.
- `warmup`: الافتراضي 10.
- `iters`: الافتراضي 100.

## Checks

### 1. Warmup
Run the model `warmup` times without timing. Catches first-forward JIT compilation and cold cache effects.

### 2. Synchronisation
For `cuda`, call `torch.cuda.synchronize()` before and after each timed forward pass.
For `mps`, call `torch.mps.synchronize()`.

### 3. Timer
Use `time.perf_counter()` for wall-clock measurement. Convert to milliseconds.

### 4. Percentiles
Sort the full list of timings. Report `p50, p90, p95, p99, mean, std`.

### 5. Memory
For `cuda`, call `torch.cuda.max_memory_allocated()` after the run and subtract any baseline.
For `cpu`, use `tracemalloc` or `psutil.Process().memory_info().rss` before and after.

### 6. Batch-size sweep
Optionally repeat the benchmark for `batch_size in [1, 4, 16, 32]` to reveal throughput vs latency tradeoffs.

## Output template

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

## Rules

- قم دائمًا بإجراء عملية الإحماء؛ لا تثق أبدًا في التوقيت الأول.
- النسب المئوية، ليست متوسطة - يمكن للقيمة المتطرفة الواحدة مضاعفة المتوسط ​​ولكنها بالكاد تتحرك p50.
- استخدم نفس input_shape للإنتاج؛ الكمون على 224x224 ليس الكمون على 384x384.
- بالنسبة إلى CUDA، لا تحذف أبدًا `torch.cuda.synchronize()`؛ الأرقام لا معنى لها بدونها.
- قم بتسجيل إصدار الشعلة، وإصدار CUDA، واسم الجهاز إلى جانب الأرقام - ولن تكون قابلة للمقارنة بخلاف ذلك.
