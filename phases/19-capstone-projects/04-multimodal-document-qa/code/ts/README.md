# Capstone 04 - Multimodal Document QA (TypeScript)

الهيكل العظمي للعارض الذي يُرجع صورة الصفحة URL بالإضافة إلى قائمة JSON من الحدود المقتبسة
صناديق للوثيقة. تتضمن الاستجابة HTML نصًا صغيرًا متراكبًا
الذي يرسم المناطق المذكورة أعلى صورة الصفحة. أزواج مع بايثون
pipخط في `../main.py`.

## Layout

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # entrypoint, demo + HTTP server
    server.ts       # hono app, /health, /, /document/:id
    fixtures.ts     # 10-K table + Nature figure fixtures
    render.ts       # HTML index + per-document overlay renderer
    types.ts        # DocumentFixture, EvidenceRegion, BoundingBox
  tests/
    fixtures.test.ts
    render.test.ts
    server.test.ts
```

## Run

```bash
npm install
npm run typecheck
npm test
npm start          # one self-check pass, exits 0
npm run serve      # interactive HTTP server on 127.0.0.1:<port>
```

يختار الخادم التفاعلي منفذًا مجانيًا عند إلغاء تعيين `PORT` ويطبع الملف
تم اختيار URL على stdout. قم بزيارة `/` للحصول على الفهرس، `/document/10k-acme-2025` للحصول على
تراكب العرض التوضيحي، أو قم بتعيين `accept: application/json` للحصول على الاستجابة المنظمة.

## Tests

`node --test` عداء عبر tsx. تغطي الاختبارات البحث عن التركيبات (إيجابية + سلبية)،
HTML الهروب للشخصيات الخمس المعادية، توثيق HTML هيكل الحمولة،
وطرق الشرف (200، 404، التفاوض على المحتوى).
