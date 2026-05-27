# تعليمات الاستخدام

هذا الملف يشرح كيفية تشغيل المشروع ومراجعة النسخة العربية والمحافظة على تطابقها مع المصدر.

## أين توجد النسختان

- المشروع الأصلي: `/home/youssef-ayad/ai-engineering-from-scratch`
- النسخة العربية: `/home/youssef-ayad/ai-engineering-from-scratch-ar`

الترتيب الداخلي للملفات واحد في النسختين، لذلك أي تعديل مهم في المصدر يجب أن يُعكس في النسخة العربية.

## تشغيل الموقع محليًا

لتشغيل الموقع من النسخة العربية:

```bash
cd /home/youssef-ayad/ai-engineering-from-scratch-ar/site
python3 -m http.server 8000
```

ثم افتح:

- `http://127.0.0.1:8000/index.html`
- `http://127.0.0.1:8000/lesson.html`
- `http://127.0.0.1:8000/catalog.html`

إذا أردت تشغيله من جذر المشروع بدل `site/`:

```bash
python3 -m http.server 8000 --directory /home/youssef-ayad/ai-engineering-from-scratch-ar
```

وفي هذه الحالة تصبح الصفحات تحت المسار:

- `http://127.0.0.1:8000/site/index.html`
- `http://127.0.0.1:8000/site/lesson.html`

## فحوصات السلامة

قبل الاعتماد على أي نسخة جديدة، شغّل الفحوصات التالية من جذر المشروع الأصلي:

```bash
python3 scripts/audit_lessons.py --strict
python3 scripts/check_readme_counts.py
python3 scripts/lesson_run.py --strict
```

هذه الأوامر تتحقق من:

- بنية الدروس
- تطابق أعداد README مع `catalog.json`
- سلامة ملفات Python نحويًا

## إعادة بناء النسخة العربية

إذا غيّرت ملفات المصدر وتريد إعادة توليد المرآة العربية، شغّل:

```bash
python3 /home/youssef-ayad/ai-engineering-from-scratch/scripts/translate_arabic_mirror.py
```

أو إذا كنت تستخدم المشغّل المؤقت:

```bash
python3 /tmp/translate_phases_fast.py
```

بعد الانتهاء، أعد فحص التطابق بين المصدر والنسخة العربية.

## متابعة التقدم

لمراقبة عدد الملفات المترجمة في النسخة العربية:

```bash
find /home/youssef-ayad/ai-engineering-from-scratch-ar -type f -name "*.md" | wc -l
```

وللمتابعة المستمرة:

```bash
watch -n 10 'find /home/youssef-ayad/ai-engineering-from-scratch-ar -type f -name "*.md" | wc -l'
```

## قواعد عمل مهمة

- لا تعدّل ملفات المرآة العربية يدويًا إذا كان الهدف هو الحفاظ على التطابق مع المصدر.
- أي تغيير في ملفات المحتوى يجب أن يمر على الفحوصات الثلاثة السابقة.
- ملفات `site/data.js` و`site/lesson.html` جزء من واجهة الموقع ويجب أن تبقيا موجودتين في النسختين.

