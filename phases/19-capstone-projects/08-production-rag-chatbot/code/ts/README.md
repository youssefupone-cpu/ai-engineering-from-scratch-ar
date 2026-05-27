# Capstone 08 - Production RAG Chatbot (TypeScript)

هيكل عظمي للدردشة UI يبث استجابة مرتكزة على الاقتباس عبر الخادم المرسل
الأحداث. أزواج مع خط بايثون pipe في `../main.py`. حالة المحادثة تعيش
في خريطة قيد التشغيل يتم مفتاحها بواسطة `sessionId`، بحيث يمكن تشغيل معرف الجلسة نفسه
حوارات متعددة المنعطفات.

## Layout

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts        # entrypoint, demo + HTTP server
    server.ts      # hono app, /, /chat/stream (SSE), /sessions, /health
    session.ts     # SessionStore (Map<sessionId, Session>)
    stream.ts      # SSE frame encoder + parser + mock retrieval + tokenizer
    types.ts        # Session, Turn, Citation, KbEntry, SseEvent
  tests/
    session.test.ts
    stream.test.ts
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

يختار الخادم التفاعلي منفذًا مجانيًا عند عدم تعيين `PORT`، ويقوم بتحميل الدردشة
HTML العميل على `/`، والبث عبر `GET /chat/stream?sessionId=...&q=...`. ال
يستخدم العميل التجريبي `EventSource` ويستمع إلى `session`، `citations`، `token`،
و `done` الأحداث.

## Tests

`node --test` عداء عبر tsx. التغطية:

- SessionStore: إنشاء، بحث، إلحاق، قائمة، عدم السماح بالمعرف المفقود.
- SSE التشفير + المحلل اللغوي ذهابًا وإيابًا؛ تعزيز الاسترجاع عن طريق علامة الاختصاص؛ احتياطي الرمز المميز + الذيل "انظر أيضًا".
- الخادم: `/`، `/health`، `/chat/stream` المسار السعيد (جلسة + استشهادات + الرمز المميز + تم)، 400 على q المفقود، استمرار الجلسة متعددة المنعطفات، `/sessions` القائمة.
