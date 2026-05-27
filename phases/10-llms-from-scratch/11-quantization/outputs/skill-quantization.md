---
name: skill-quantization
description: Choose the right quantization strategy for deploying LLMs based on hardware, quality, and latency constraints
version: 1.0.0
phase: 10
lesson: 11
tags: [quantization, inference, deployment, optimization, fp8, int4, int8, gptq, awq, gguf]
---

# Quantization Decision Framework

عند نشر نموذج لغة، استخدم إطار العمل هذا لتحديد تنسيق الأرقام الصحيح وطريقة القياس الكمي واستراتيجية التحقق من الجودة.

## Input Requirements

تقديم:
- **النموذج** (الاسم، عدد المعلمات، الدقة الأصلية)
- **الأجهزة المستهدفة** (طراز GPU/VRAM، CPU، Apple Silicon، جهاز حافة)
- **هدف زمن الوصول** (الرموز المميزة في الثانية، الوقت حتى الرمز المميز الأول)
- **حد أدنى للجودة** (الحد الأقصى المقبول لزيادة الارتباك، دلتا المعيارية)
- **نمط العرض** (حجم الدفعة، الحد الأقصى لطول السياق، المستخدمين المتزامنين)

## Quick Selection

| حالتك | تنسيق | الطريقة | فقدان الجودة المتوقع |
|---------------|--------|-------|---------------------|
| H100 GPU، الحد الأقصى للإنتاجية | FP8 E4M3 | صب H100 أصلي | < 0.1% |
| A100/A10، تحتاج إلى إنتاجية 2x | INT8 | LLM.int8() أو SmoothQuant | < 0.5% |
| مفرد 24 جيجا GPU موديل 70B | INT4 | AWQ أو GPTQ | 1-3% |
| ماك بوك / أبل سيليكون | INT4 GGUF | Q4_K_M عبر llama.cpp | 1-2% |
| جهاز محمول / حافة | INT4 أو INT3 | QAT+ خاص بالجهاز | 2-5% |
| أقصى ضغط، بعض الخسارة OK | INT2 | QuiP# أو AQLM | 5-15% |
| التدريب (الدقة المختلطة) | BF16 + FP32 تراكم | دعم الإطار الأصلي | 0% |

## Precision Selection by Component

لا يجب أن تحصل جميع الموترات على نفس المعاملة.

| مكون | الحد الأدنى الآمن | موصى به | تجنب |
|-----------|--------------------|-------|------|------|
| FFN الأوزان | INT4 | INT4 (AWQ/GPTQ) | INT2 بلا QAT |
| اوزان الانتباه | INT4 | INT8 أو FP8 | INT2 |
| طبقة التضمين | INT8 | FP16 (احتفظ بالأصل) | INT4 |
| رأس الإخراج | INT8 | FP16 (احتفظ بالأصل) | INT4 |
| KV مخبأ | FP8 | FP8 أو INT8 | INT4 في سياق طويل |
| انتبه لوgits | FP16 | FP16 أو BF16 | INT8 |
| التنشيط (الاستدلال) | INT8 | FP8 أو INT8 | INT4 |

## Method Comparison

### GPTQ
- **When:** GPU inference, you want a Hugging Face-compatible model
- **Calibration data:** 128 examples, 2048 tokens each
- **Time:** 30-60 minutes for 70B on A100
- **Tooling:** `auto-gptq`, `exllama`, `exllamav2`
- **Strength:** Well-tested, huge model zoo on Hugging Face
- **Weakness:** Slower than AWQ to apply, slightly lower quality than AWQ on some models

### AWQ
- **When:** GPU inference, you want best quality-per-bit
- **Calibration data:** 128 examples
- **Time:** 15-30 minutes for 70B on A100
- **Tooling:** `autoawq`, `vLLM` (native support)
- **Strength:** Best INT4 quality, fast to apply, vLLM integration
- **Weakness:** Smaller model zoo than GPTQ

### GGUF
- **When:** CPU inference, Apple Silicon, llama.cpp ecosystem
- **Variants:** Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K, Q8_0, F16
- **Recommended default:** Q4_K_M (best quality/size balance)
- **Tooling:** `llama.cpp`, `ollama`, `LM Studio`
- **Strength:** Self-contained files, mixed precision, massive ecosystem
- **Weakness:** Not optimal for GPU (designed for CPU/Metal)

### SmoothQuant
- **When:** INT8 on GPU, need both weight and activation quantization
- **Key idea:** Migrate quantization difficulty from activations to weights via per-channel scaling
- **Tooling:** `smoothquant`, `TensorRT-LLM`
- **Strength:** Enables W8A8 (both weights and activations in INT8) for 2x speedup
- **Weakness:** INT8 only, does not extend to INT4

## Quality Validation Protocol

بعد القياس الكمي، تحقق من الصحة قبل النشر:

1. **اختبار الحيرة.** قم بالحساب على WikiText-2 أو مجموعة النطاق الخاصة بك. دلتا <0.5 ممتاز، 0.5-1.0 جيد،> 2.0 يمثل مشكلة.

2. **المسح المعياري.** تشغيل MMLU (عام)، GSM8K (رياضيات)، HumanEval (كود). تعتبر الرياضيات والتعليمات البرمجية أكثر حساسية لفقدان الدقة.

3. **مقارنة المخرجات.** قم بإنشاء 100 رد من النموذج الأصلي والنموذج الكمي. استخدم LLM-كحكم لحساب معدل الفوز. الهدف: النموذج الكمي يفوز أو يتعادل في أكثر من 90% من المطالبات.

4. **قياس زمن الوصول.** قم بقياس الرموز المميزة/الثانية عند حجم الدُفعة 1 وحجم الدُفعة المستهدف. التحقق من أن التسريع يبرر تكلفة الجودة.

5. **اختبار السياق الطويل.** إذا كنت تخدم سياقات طويلة (> 4K من الرموز المميزة)، فاختبر الحد الأقصى لطول السياق. KV أخطاء تكميم ذاكرة التخزين المؤقت مركبة مع طول التسلسل.

## Memory Budget Calculator

```
Weight memory (GB) = parameters (B) * bits / 8 / 1.073741824
KV cache per token (MB) = 2 * num_layers * d_model * bits / 8 / 1048576
KV cache for context (GB) = kv_per_token * max_context_length / 1024
Activation memory (GB) ~ 1-4 GB (relatively constant, depends on batch size)
Total = weight_memory + kv_cache + activation_memory + overhead (10-20%)
```

مثال لـ Llama 3 70B عند INT4، سياق 32K:
- الأوزان: 70 ب * 4 / 8 / 1.07 = 32.6 GB
- KV ذاكرة التخزين المؤقت (FP16): 2 * 80 * 8192 * 16 / 8 / 1e9 * 32768 = ~40 GB
- KV ذاكرة التخزين المؤقت (FP8): ~20 GB
- الإجمالي مع FP8 KV: ~55 GB (يناسب جهازًا واحدًا بسعة 80 جيجابايت A100)

## Common Mistakes

| خطأ | لماذا يفشل | إصلاح |
|---------|------------|-----|
| تكميم طبقة التضمين إلى INT4 | تعمل الطبقة الأولى على تضخيم الأخطاء من خلال النموذج بأكمله | احتفظ بالتضمينات عند FP16 أو INT8 |
| استخدام المقاييس لكل موتر لـ INT4 | صف واحد خارجي يدمر الدقة لجميع الصفوف | استخدم المقاييس لكل قناة أو لكل مجموعة |
| عدم المعايرة GPTQ/AWQ | عوامل القياس خاطئة بدون بيانات تمثيلية | استخدم 128 مثالاً من المجال الخاص بك |
| نفس عرض البت لجميع الطبقات | الطبقات الأولى/الأخيرة أكثر حساسية | دقة مختلطة: بتات أعلى للأول/الأخير |
| تكميم ذاكرة التخزين المؤقت KV في سياق طويل جدًا | تتراكم الأخطاء بشكل تربيعي مع طول التسلسل | استخدم FP8 لـ KV ذاكرة تخزين مؤقت، وليس INT4 |
| تخطي التحقق من الجودة | بعض النماذج يتم قياسها بشكل سيء (خاصة عند الحدود) | قم دائمًا بتشغيل الحيرة + تقييمات المهام |

## Deployment Recipes

### Recipe 1: vLLM with AWQ (GPU server)
```
pip install vllm autoawq
vllm serve model-awq --quantization awq --dtype half --max-model-len 8192
```

### Recipe 2: llama.cpp with GGUF (MacBook)
```
./llama-server -m model.Q4_K_M.gguf -c 4096 -ngl 99
```

### Recipe 3: TensorRT-LLM with FP8 (H100)
```
trtllm-build --model_dir model --output_dir engine --dtype float16 --use_fp8
```
