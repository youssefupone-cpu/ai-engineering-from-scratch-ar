# تعليمات الاستخدام

هذه نسخة عربية مخصصة مبنية على المستودع الأصلي:

- الأصل: [rohitg00/ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
- المرآة العربية: [youssefupone-cpu/ai-engineering-from-scratch-ar](https://github.com/youssefupone-cpu/ai-engineering-from-scratch-ar)

الهدف من هذه النسخة هو الحفاظ على نفس البنية التعليمية والمحتوى البرمجي مع تقديم الشرح والملاحظات بالعربية.

## تشغيل الموقع محليًا

من جذر المشروع:

```bash
python3 -m http.server 8000 --directory .
```

ثم افتح:

- `http://127.0.0.1:8000/site/index.html`
- `http://127.0.0.1:8000/site/lesson.html`
- `http://127.0.0.1:8000/site/catalog.html`

## فحص سلامة المحتوى

قبل الاعتماد على أي تحديث، شغّل الفحوصات التالية من الجذر:

```bash
python3 scripts/audit_lessons.py --strict
python3 scripts/build_catalog.py
python3 scripts/check_readme_counts.py
python3 scripts/lesson_run.py --strict
```

هذه الأوامر تتحقق من:

- بنية الدروس والملفات
- تطابق أعداد العناوين مع المخرجات المولدة
- سلامة ملفات Python والتجارب التعليمية

## إعادة توليد المرآة العربية

إذا أردت إعادة توليد النسخة العربية من المصدر الأصلي، استخدم:

```bash
python3 scripts/translate_arabic_mirror.py
```

أو شغّل المسرّع المؤقت إذا كان متاحًا:

```bash
python3 /tmp/translate_phases_fast.py
```

## متابعة التقدم

لمعرفة عدد ملفات Markdown في المرآة:

```bash
find . -type f -name "*.md" | wc -l
```

وللمراقبة المستمرة:

```bash
watch -n 10 'find . -type f -name "*.md" | wc -l'
```

## ملاحظات مهمة

- هذه المرآة العربية مخصّصة للقارئ العربي، لكنها تبقي الروابط والبنية والمسارات متوافقة مع المشروع الأصلي.
- إذا تغيّر المصدر الأصلي، أعد التوليد ثم أعد تشغيل الفحوصات السابقة قبل النشر.
- في حالة تشغيل الموقع من مجلد آخر، غيّر المسار في أمر `python3 -m http.server` بما يناسب مكان المستودع لديك.

