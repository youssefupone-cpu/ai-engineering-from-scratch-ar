---
name: skill-concept-prompt-designer
description: Turn user utterances into well-formed SAM 3 concept prompts with splitting, disambiguation, and fallbacks
version: 1.0.0
phase: 4
lesson: 24
tags: [sam3, open-vocab, prompt-engineering, segmentation]
---

# مصمم المفهوم الفوري
تعتمد دقة SAM 3 بشكل كبير على كيفية صياغة الفكرة. تعمل هذه المهارة على تطبيع عبارات المستخدم ذات الشكل الحر إلى مطالبات يتعامل معها SAM 3 بشكل جيد.
##متى يستخدم
- إنشاء UI يقبل استعلامات كائنات اللغة الطبيعية.
- الكشف عن SAM 3 من خلال API حيث يرسل المتصلون الرئيسيون الجمل.
- تصحيح أخطاء التطابقات الضعيفة لـ SAM 3 - غالبًا ما يكون الموجه مشوهًا، وليس النموذج.
## المدخلات
- `utterance`: سلسلة مستخدم أولية.
- `context`: تلميح نطاق اختياري (على سبيل المثال، "المراقبة"، "الطبية"، "البيع بالتجزئة").
- `max_concepts`: الحد الأقصى من المفاهيم التي يمكن استخلاصها من كل كلمة؛ الافتراضي 5.
## القواعد SAM 3 يفضل
- **العبارات الاسمية القصيرة، وليس الجمل.** `"cat"` يفوز على `"there is a cat"`.
- **الأسماء الخرسانية.** `"skateboard"` يفوز على `"thing to ride on"`.
- **المعدلات مباشرة قبل الاسم.** `"red car"` يفوز على `"car that is red"`.
- **الأحرف الصغيرة.** SAM 3 قوية ولكنها أفضل قليلاً من الناحية التجريبية في المدخلات بالأحرف الصغيرة.
- **المفرد أو الجمع.** كلاهما يعمل؛ يساعد الجمع عند توقع وجود مثيلات متعددة.
## الخطوات
1. **الترميز بواسطة الفواصل المشتركة** — الفاصلة، والفاصلة المنقوطة، و"و"، و"أو"، و"&".
2. **إسقاط بادئات الحشو** — "بحث"، "أرني"، "مقطع"، "اكتشاف"، "تحديد موقع"، "a"، "an"، "the".
3. **احتفظ بمعدلات حروف الجر** فقط إذا كانت مرئية — `"striped red umbrella"` نعم، `"umbrella from yesterday"` لا (`"from yesterday"` ليس موجودًا في الصورة).
4. **إزالة الغموض عن التصادمات** باستخدام `context` الاختياري:
   - `"window"` في سياق المراقبة -> `"building window"`.
   - `"window"` في السياق الطبي -> الخطأ غالبًا؛ أقترح توضيح المستخدم.
5. **الرجوع** إلى السلسلة الدقيقة إذا لم ينتج عن التقسيم أي مفاهيم *و* يحتوي الكلام على اسم ملموس واحد على الأقل. إذا لم يكن من الممكن استخراج اسم ملموس، فلا تصدر مفهومًا - قم بإرجاع التحذيرات فقط واطلب من المستخدم التوضيح (راجع القواعد).
6. **الحد الأقصى عند `max_concepts`.** إذا تم استخراج مفاهيم أكثر مما طلبه المتصل، فاحتفظ بـ `max_concepts` الأول في ترتيب النطق وأرسل الباقي تحت `dropped` مع السبب `"exceeded max_concepts"`. يؤدي هذا إلى إبقاء زمن الوصول محدودًا عندما يقوم المستخدم بلصق تعداد طويل.
## تنسيق الإخراج
```
[designed prompts]
  utterance:    <original>
  concepts:     ["concept_1", "concept_2", ...]
  dropped:      ["filler_1", ...]
  warnings:     ["concept too abstract", "may match many classes", ...]

[sam3 calls]
  For each concept run: sam3.detect(image, concept)
  Merge outputs with distinct concept tags per detection.
```

## أمثلة
```
in:  "can you find me a cat or two dogs?"
out: ["cat", "dogs"]
dropped: ["can you find me", "a", "or two", "?"]
note: "dogs" kept plural because the utterance says "two dogs" — plural hint preserved.

in:  "segment the big red truck and the blue sedan"
out: ["big red truck", "blue sedan"]
dropped: ["segment", "the", "and"]

in:  "thing near the door"
out: ["door"]
warnings: ["'thing' is too abstract for SAM 3; fell back to 'door'"]

in:  "striped red umbrella, green hat, pink balloon"
out: ["striped red umbrella", "green hat", "pink balloon"]
```

## قواعد
- لا تمرر أبدًا جملًا أطول من 8 كلمات إلى SAM 3 - تنخفض الدقة فوق ذلك.
- عندما لا يحتوي الكلام على أسماء ملموسة قابلة للاستخراج، لا تقم بتشغيل SAM 3؛ أعد التحذيرات واطلب التوضيح.
- لا تقسم علامات الترقيم داخل السلاسل المقتبسة؛ احتفظ بـ `"black and white cat"` كمفهوم واحد إذا تم اقتباسه.
- قم دائمًا بتسجيل الكلام الأصلي والمفاهيم المشتقة لتصحيح أخطاء الإنتاج.