# Capstone 19/02 — RAG over Codebase (TypeScript)

ملف متعدد TypeScript بحث الكود API للاسترجاع المختلط pipeline
الموصوفة في `../docs/en.md`. غير متصل، حتمية، عينة مكونة من ستة أجزاء،
node:http خلف معالج جلب hono.

## Layout

```text
src/
  index.ts        entry point; boots node:http + self-probe + exits 0
  server.ts       hono routes (/healthz, /query) with zod-validated POST body
  retrieval.ts    runQuery + RRF merge over dense and BM25
  index_store.ts  FNV-1a hash embedder, cosine, field-weighted BM25
  corpus.ts       six-chunk sample (uploader / auth / client / catalog)
  types.ts        Chunk, RankedChunk, QueryResponse, anchor()
tests/
  index_store.test.ts
  retrieval.test.ts
  server.test.ts
```

## Run

```bash
npm install
npm start                # boots api, probes three queries, exits 0
npm start -- --serve     # keep server up; ctrl-c to stop
npm test                 # node --test runner via tsx
npm run typecheck        # tsc --noEmit
```

يؤكد المسار `npm start` غير التفاعلي على أن `/healthz` يُرجع 200 و
أن كل استعلام بحثي يُرجع اقتباسًا واحدًا على الأقل. الطرق:

- `GET /healthz` — ترجع `{ok, corpus}`.
- `GET /query?q=...` — يقوم بتشغيل استعلام مختلط.
- `POST /query` — JSON `{q, topK?}`، تم التحقق من صحته بواسطة zod (`topK` بحد أقصى 50).
