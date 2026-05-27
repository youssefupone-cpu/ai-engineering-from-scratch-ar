# Contributing

الدروس والترجمات والإصلاحات والمخرجات - نرحب بالجميع. مساهمة واحدة لكل سحب
يحافظ الطلب على المراجعات بسرعة ويسمح بإحصاء المساهمين وعمل الائتمان
بشكل صحيح.

## Important: the README and ROADMAP feed the website

`site/build.js` يوزع `__TERM_0__.md` و`__TERM_1__.md` و`glossary/terms.md` إلى
إنشاء `site/data.js`. يجب أن يظل النموذجان سليمين في أي طلب سحب
يمس تلك الملفات:

- رؤوس المراحل في أي من دروس `### Phase N: Name \`X\`` form or `<التفاصيل><summary><b>المرحلة N — الاسم</b>... <code>X الدروس</code>... <em>الوصف</em></summary>` form.
- Lesson tables with the column shape `| # | الدرس | اكتب | لانج |` (or `| # | مشروع | يجمع | Lang |` for capstone tables). The `Lang` column accepts plain text (`Python, TypeScript`) or the legacy emoji flags (`🐍 🟦 🦀 🟣 ⚛️`); both are parser-equivalent.
- __TERM_1__ status glyphs (`✅`, `🚧`, `⬚`) على رؤوس المراحل وصفوف الدرس. لا تستبدلها بنص - يقوم المحلل اللغوي بإخراج الأحرف الدقيقة.

قم بتشغيل `__TERM_0__ site/build.js` بعد تحرير تلك الملفات؛ __الكود_1__
يجب أن يُظهر تغيير الطابع الزمني فقط إذا كان تعديلك آمنًا من الناحية الهيكلية.

## Ways to Contribute

### 1. Add a New Lesson

كل درس موجود في `phases/__TERM_0__-phase-name/__TERM_1__-lesson-name/` بهذه البنية:

```
NN-lesson-name/
├── code/           At least one runnable implementation
├── notebook/       Jupyter notebook for experimentation (optional)
├── docs/
│   └── en.md       Lesson documentation (required)
└── outputs/        Prompts, skills, or agents this lesson produces (if applicable)
```

**تنسيق مستند الدرس** (`en.md`):

```markdown
# Lesson Title

> One-line motto — the core idea in one sentence.

## The Problem

Why does this matter? What can't you do without this?

## The Concept

Explain with diagrams, visuals, and intuition. Code comes later.

## Build It

Step-by-step implementation from scratch.

## Use It

Now use a real framework or library to do the same thing.

## Ship It

The prompt, skill, agent, or tool this lesson produces.

## Exercises

1. Exercise one
2. Exercise two
3. Challenge exercise
```

### 2. Add a Translation

قم بإنشاء ملف جديد في مجلد `docs/` الخاص بأي درس:

```
docs/
├── en.md    (English — always required)
├── zh.md    (Chinese)
├── ja.md    (Japanese)
├── es.md    (Spanish)
├── hi.md    (Hindi)
└── ...
```

احتفظ بنفس هيكل النسخة الإنجليزية. ترجمة المحتوى، وليس التعليمات البرمجية.

### 3. Add an Output

إذا كان من المفترض أن ينتج عن الدرس مطالبة أو مهارة أو وكيل أو خادم MCP قابل لإعادة الاستخدام:

1. قم بإنشائه في مجلد الدرس `outputs/`
2. أضف مرجعًا في فهرس المستوى الأعلى `outputs/`

** التنسيق الفوري: **

```markdown
---
name: prompt-name
description: What this prompt does
phase: 14
lesson: 01
---

[System prompt or template here]
```

**شكل المهارة:**

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: 14
lesson: 01
tags: [agents, loops]
---

[Skill content here]
```

### 4. Fix Bugs or Improve Existing Lessons

- إصلاح الكود الذي لا يعمل
- تحسين الشروحات
- إضافة رسوم بيانية أفضل
- تحديث المعلومات القديمة

### 5. Add Exercises or Projects

نرحب دائمًا بمزيد من التمارين والمشاريع، خاصة تلك التي تربط بين مراحل متعددة.

## Guidelines

- **يجب تشغيل التعليمات البرمجية.** يجب تنفيذ كل ملف تعليمات برمجية بدون أخطاء في التبعيات المدرجة.
- **لا توجد تعليقات في الكود.** يجب أن يكون الكود واضحًا بذاته. استخدم المستندات للتوضيح.
- **أفضل لغة للوظيفة.** لا تجبر لغة Python على أن يكون TypeScript أو Rust هو الخيار الأفضل.
- **البناء من الصفر أولاً.** قم دائمًا بتنفيذ المفهوم من المبادئ الأولى قبل عرض إصدار إطار العمل.
- **حافظ على التطبيق العملي.** النظرية تخدم الممارسة وليس العكس.
- **لا يوجد AI هراء.** اكتب مثل الإنسان. كن مباشرا. قطع حشو.

## Pull Request Process

1. شوكة المستودع
2. إنشاء فرع الميزات (`__TERM_0__ checkout -b add-lesson-phase3-gradient-descent`)
3. قم بإجراء التغييرات الخاصة بك
4. تأكد من تشغيل كافة التعليمات البرمجية
5. أرسل طلب سحب مع وصف واضح

## Code of Conduct

انظر [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). كن لطيفًا، وكن مفيدًا، وكن بناءًا.

## Style

- النثر المباشر. قطع حشو. قم بمطابقة نغمة الدليل وليس النسخة التسويقية.
- لا توجد رموز تعبيرية زخرفية في العناوين. أعلام الرموز التعبيرية لعمود لانج هي واحدة استثناء وفقط لأن المحلل اللغوي يعينهم.
- يعمل الكود كما هو مع التبعيات المدرجة في الدرس.
- البناء من الصفر أولاً، والإطار ثانياً.
