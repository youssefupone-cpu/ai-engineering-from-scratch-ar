---
name: skill-residual-block-reviewer
description: Review a PyTorch residual block for skip-connection correctness, BN placement, activation order, and shape alignment
version: 1.0.0
phase: 4
lesson: 3
tags: [computer-vision, resnet, code-review, pytorch]
---

# مراجع الكتلة المتبقية
مراجع مركز لأي PyTorch `nn.Module` يدعي تنفيذ كتلة متبقية. يلتقط الأخطاء الأربعة التي تمثل تقريبًا كل عملية إعادة كتابة معطلة لـ ResNet.
##متى يستخدم
- قام شخص ما بكتابة BasicBlock أو اختناق مخصص وكانت الخسارة NaN أو توقفت الدقة.
- أنت تقوم بنقل كتلة من إطار عمل إلى آخر وتريد التحقق من التكافؤ.
- أنت تقوم بمراجعة PR الذي يغير الأجزاء الداخلية لـ ResNet (التنشيط المسبق، والضغط المثير، والتنعيم).
- يتم شحن النموذج بشكل جيد عند إدخال بحجم CIFAR ولكنه يتعطل عند دقة ImageNet لأن الاختصار غير صحيح.
## المدخلات
- تعريف فئة PyTorch، إما كنص مصدر أو مسار قابل للاستيراد.
- اختياري `variant`: `basic` | __الكود_2__ | __الكود_3__ | __الكود_4__.
## أربعة شيكات
### 1. محاذاة شكل الاختصار
بالنسبة لأي كتلة تحتوي على `stride != 1` أو `in_channels != out_channels`، يجب أن يكون مسار الاختصار **يجب** أن يكون وحدة نمطية مطابقة للشكل - عادةً تحويل 1x1 بالإضافة إلى BN. يعد `nn.Identity()` العاري في هذه الحالة خطأً مضمونًا في عدم تطابق الشكل في وقت لاحق.
التشخيص:```
[shortcut]
  detected:  nn.Identity | 1x1 Conv + BN | 1x1 Conv + BN + ReLU | other
  required:  shape-matching Conv if (stride != 1 or in_c != out_c) else Identity
  verdict:   ok | wrong | unnecessarily heavy
```

### 2. BN الموضع بالنسبة لعملية الإضافة
يجب أن تحدث الإضافة `out + shortcut(x)` **قبل** ReLU النهائي (ما بعد التنشيط، ResNet الأصلي) أو ReLU النهائي يجب أن يكون غائبًا تمامًا (التنشيط المسبق ResNet v2). الكتلة التي تطبق ReLU في الفرع الرئيسي ثم تضيف اختصارًا أوليًا تنتج نطاق تنشيط غير متماثل يضر بالتدريب.
التشخيص:```
[activation order]
  pattern:  post-act (conv-BN-ReLU-conv-BN-add-ReLU) | pre-act (BN-ReLU-conv-BN-ReLU-conv-add) | other
  verdict:  ok | suspect
```

### 3. التحيز في طبقات التحويل
يجب أن تحتوي التحويلات التي تتبعها BatchNorm مباشرة على `bias=False`. يقوم الإصدار التجريبي من BN بالفعل بتحديد معلمات الانحياز، وبالتالي يؤدي انحياز التحويل الإضافي إلى إهدار المعلمات ويمكن أن يؤدي إلى إبطاء التقارب.
التشخيص:```
[bias]
  convs with BN and bias=True: <count>
  recommended fix: set bias=False on those layers
```

### 4. ReLU في المكان وAutograd
`nn.ReLU(inplace=True)` الموجود على الموتر الذي سيتم إضافته إلى الاختصار يحل محل القيم التي قد تظل مطلوبة للإضافة المتبقية. ضع علامة على أي `inplace=True` لا تتبعه طبقة تنتج موترًا جديدًا قبل الإضافة.
التشخيص:```
[in-place]
  risky inplace ops: <list>
  fix: inplace=False before the residual add
```

## تقرير
```
[block-review]
  variant:       basic | bottleneck | preact | se | other
  shortcut:      ok | wrong | heavy
  activation:    ok | suspect
  bias-bn:       ok | <N> convs need bias=False
  in-place:      ok | <N> risky ops
  summary:       one sentence
```

## قواعد
- لا تعيد كتابة الكتلة. تقرير فقط.
- إذا كانت الكتلة صحيحة، قل `ok` في كل مكان وتوقف. لا توجد اقتراحات.
- إذا كانت هناك عدة أشياء خاطئة، فقم بإدراجها بالترتيب أعلاه (الاختصار أولاً لأنه السبب الأكثر شيوعًا للحوادث).
- لا تضع علامة أبدًا على متغير التنشيط المسبق أو الضغط المثير على أنه خاطئ عندما يحدده المستخدم.