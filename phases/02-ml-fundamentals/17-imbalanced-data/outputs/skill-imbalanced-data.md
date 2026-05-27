---
name: skill-imbalanced-data
description: Decision checklist for handling imbalanced classification problems
version: 1.0.0
phase: 2
lesson: 17
tags: [imbalanced-data, smote, class-weights, threshold-tuning, evaluation]
---

# Imbalanced Data Strategy

قائمة مرجعية للقرارات للتعامل مع التصنيف غير المتوازن. اتبع هذا التسلسل لاختيار النهج الصحيح لمشكلتك.

## Step 1: Measure the imbalance

- عد العينات لكل فئة
- حساب نسبة الخلل (الأغلبية / الأقلية)
- خفيف: النسبة < 3:1 (على سبيل المثال، 70/30)
- معتدلة: النسبة 3:1 إلى 20:1 (على سبيل المثال، 95/5)
- شديدة: النسبة > 20:1 (على سبيل المثال، 99/1)

## Step 2: Pick the right metric

تفضل الدقة/الاستدعاء/F1 على الدقة لمجموعات البيانات غير المتوازنة. اختر بناءً على مشكلتك:

| الوضع | المقياس الأساسي | Secondary متري |
|-----------|-------------|-----------------|
| ضياع الإيجابيات مكلف جداً (احتيال، مرض) | أذكر | F2 النتيجة |
| الإنذارات الكاذبة مكلفة (مرشح البريد العشوائي، التوصيات) | الدقة | F0.5 النتيجة |
| كلاهما مهمان بشكل متساوٍ تقريبًا | F1 النتيجة | MCC |
| بحاجة إلى مقياس تصنيف واحد | AUPRC | AUC - ROC |
| تحتاج إلى المقارنة بين مجموعات البيانات | MCC | AUPRC |

## Step 3: Choose a rebalancing strategy

### By imbalance severity

| خلل | المحاولة الأولى | المحاولة الثانية | تجنب |
|-----------|-----------|------------|-------|
| خفيف (< 3:1) | أوزان الطبقة | ضبط العتبة | الإفراط في أخذ العينات (غير ضروري) |
| معتدل (3:1 إلى 20:1) | SMOTE+أوزان الصنف | ضبط العتبة في الأعلى | Undersampling (فقد الكثير من البيانات) |
| شديد (>20:1) | SMOTE + أوزان الصنف + العتبة | مجموعة مع تعبئة متوازنة | الأخذ بعين الاعتبار وحده |

### By dataset size

| حجم مجموعة البيانات | الإستراتيجية المفضلة | السبب |
|-------------|-------------------|--------|
| < 1000 عينة | الإفراط في أخذ العينات أو SMOTE | لا أستطيع تحمل فقدان بيانات الأغلبية |
| 1,000 - 10,000 | SMOTE + ضبط العتبة | ما يكفي من عينات الأقليات لـ k-NN |
| > 10.000 | أوزان الطبقة أو الأخذ بعين الاعتبار | بيانات الأقلية سريعة وكافية |

## Step 4: Apply the technique

### Class weights (always try first)
- In sklearn: `class_weight='balanced'`
- No data modification needed
- Works with any loss-based model
- Equivalent to oversampling in expectation

### SMOTE
- Apply only to training data (never test/validation)
- Use k=5 neighbors (default)
- Combine with class weights for best results
- Watch for noisy synthetic points near the boundary

### Threshold tuning
- Train model, get predicted probabilities on validation set
- Sweep thresholds from 0.05 to 0.95
- Pick threshold maximizing your chosen metric
- Always tune on validation data, never test data

## Step 5: Validate properly

- استخدام التحقق المتبادل الطبقي (يحافظ على نسب الطبقة في كل أضعاف)
- تقرير المقاييس على مجموعة الاختبار الأصلية (غير المعاد تشكيلها).
- لا تطبق أبدًا SMOTE قبل التقسيم - فقط في طيات التدريب
- قارن مع خط الأساس "توقع الأغلبية دائمًا".

## Step 6: Common mistakes to avoid

- تطبيق SMOTE على مجموعة البيانات بأكملها قبل تقسيم التدريب/الاختبار (تسرب البيانات)
- استخدام الدقة كمقياس للتقييم
- عدم تجربة أوزان الفصل أولاً (نهج أبسط، وغالبًا ما يكون كافيًا)
- الإفراط في أخذ العينات ثم التحقق المتبادل (تسرب النقاط الاصطناعية عبر الطيات)
- تجاهل ضبط العتبة (الأداء المجاني، لا حاجة لإعادة التدريب)
- استخدام الاختزال العشوائي في مجموعات البيانات الصغيرة (يؤدي إلى التخلص من الكثير من البيانات)

## Quick Decision Tree

1. هل نسبة الخلل < 3:1؟ -> جرب أوزان الفصل فقط
2. هل مجموعة البيانات أكبر من 10000 عينة؟ -> أوزان الفئة + ضبط العتبة
3. هل مجموعة البيانات أقل من 1000 عينة؟ -> SMOTE + أوزان الفئة
4. وإلا -> SMOTE + أوزان الفئة + ضبط العتبة
5. هل ما زلت غير جيد بما فيه الكفاية؟ -> مجموعة تعبئة متوازنة
