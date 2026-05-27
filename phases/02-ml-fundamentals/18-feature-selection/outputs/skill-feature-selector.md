---
name: skill-feature-selector
description: Quick reference decision tree for choosing the right feature selection method
version: 1.0.0
phase: 2
lesson: 18
tags: [feature-selection, mutual-information, rfe, lasso, tree-importance]
---

# Feature Selection Strategy

مرجع سريع لاختيار وتطبيق طريقة اختيار الميزة الصحيحة.

## Step 1: Start with cleanup

قبل تطبيق أي طريقة، قم بإزالة الميزات غير المفيدة بشكل واضح:

- **الميزات الثابتة**: التباين = 0. قم بإزالتها.
- **ميزات شبه ثابتة**: التباين < 0.01 (أو الحد الخاص بك). قم بإزالتها.
- **ميزات مكررة**: أعمدة متطابقة. احتفظ بواحدة، واترك الباقي.
- **ID أعمدة**: فريدة لكل صف، ولا تحمل أي معلومات قابلة للتعميم. قم بإزالتها.

يستغرق هذا ثوانٍ ويمكن أن يزيل 10-30% من الميزات الموجودة في مجموعات البيانات الواقعية الفوضوية.

## Step 2: Choose a method based on your situation

### Quick Decision Tree

1. **< 50 ميزة؟ ** ابدأ بترتيب المعلومات المتبادلة. الحفاظ على أعلى ك.
2. **50 - 500 ميزة؟** استخدم حد التباين أولاً، ثم L1 (لاسو) في حالة استخدام نموذج خطي، أو أهمية الشجرة في حالة استخدام الأشجار.
3. **> 500 ميزة؟ ** طرق السلسلة: عتبة التباين -> مرشح المعلومات المتبادلة (أعلى 50%) -> RFE على الناجين.
4. **هل تحتاج إلى تفسير؟** L1 التنظيم يمنحك صفرًا/غير صفر تمامًا. أهمية الشجرة تعطي درجات مرتبة.
5. ** هل تحتاج إلى التقاط العلاقات غير الخطية؟ ** المعلومات المتبادلة أو الأهمية المبنية على الشجرة. تجنب L1 (خطي فقط).
6. **هل تحتاج إلى تفاعلات مميزة؟** RFE أو أهمية تعتمد على الشجرة. طرق التصفية تفوت التفاعلات.

### Method Reference

| الطريقة | متى تستخدم | متى يجب تجنبه |
|--------|-----------|---------------|
| عتبة التباين | دائمًا، كخطوة أولى | لا تخطي هذا أبدًا |
| معلومات متبادلة | ترتيب سريع، علاقات غير خطية | عندما تحتاج إلى اكتشاف تفاعل الميزة |
| RFE | اختيار شامل، عدد معتدل من الميزات | موديلات غالية الثمن، > 1000 ميزة |
| L1 / لاسو | النماذج الخطية، الاختيار المضمن السريع | المسائل غير الخطية، والميزات المترابطة للغاية |
| أهمية الشجرة | العلاقات غير الخطية، تفاعلات السمات | متحيزة من خلال ميزات العناصر الأساسية العالية |
| أهمية التقليب | التحقق من صحة النموذج، والفحص النهائي | بطيء جدًا بالنسبة للفحص الأولي |

## Step 3: Validate your selection

- مقارنة أداء النموذج مع الميزات المحددة مقابل جميع الميزات
- استخدم التحقق المتبادل، وليس تقسيم قطار/اختبار واحد
- إذا انخفض الأداء بأكثر من 1-2%، فربما تكون قد قمت بإزالة الميزات المفيدة
- إذا تحسن الأداء، فقد نجحت في إزالة الضوضاء

## Step 4: Handle common pitfalls

### Correlated features
- L1 arbitrarily picks one from a correlated group and zeros the others
- Compute the correlation matrix first and decide which correlated features to keep
- Tree importance spreads importance across correlated features

### Data leakage
- Fit feature selection on training data only
- Apply the same selection to test data
- In cross-validation, feature selection must happen inside each fold

### Overfitting to feature selection
- RFE with too many iterations can overfit to the training set
- Validate on held-out data, not the data used for selection
- Use stability selection (repeat on subsamples) for more robust results

## Step 5: Production checklist

- [ ] يتم تطبيق عتبة التباين كمرشح أول
- [ ] اختيار الميزة يتم تركيبه على بيانات التدريب فقط
- [ ] توثيق الميزات المختارة (الأسماء، الطريقة المستخدمة، الدرجات)
- [ ] مقارنة الأداء: الميزات المحددة مقابل جميع الميزات
- [ ] تقييم متقاطع، وليس تقييمًا منفردًا
- [ ] اختيار الميزة المدمجة في خط التدريب pipe (لا يتم ذلك يدويًا)
- [ ] مراقبة انحراف الميزات (قد تصبح الميزات المحددة قديمة)
