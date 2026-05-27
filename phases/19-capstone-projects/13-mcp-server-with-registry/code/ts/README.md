# Lesson 13 - Internal MCP Server (TypeScript)

TypeScriptنصف الكابونة. يقوم جانب Python (`code/main.py`) بشحن ملف
بوابة التسجيل والسياسة؛ هذا المشروع هو MCP النقل: المدرفلة يدويا
محدد بالسطر الجديد JSON-RPC 2.0 عبر stdio مع ثلاث أدوات وهمية للحادث. لا
`@modelcontextprotocol/sdk`; يمكنك رؤية كل بايت على السلك.

## Layout

```text
src/
  index.ts      entry: fixture demo (default) or stdio loop (--serve)
  transport.ts  stdin readline + fixture replay
  protocol.ts   initialize / tools/list / tools/call / shutdown
  tools.ts      three incident tools + executors
  types.ts      JSON-RPC + tool shapes
tests/
  protocol.test.ts  roundtrip, list shape, dispatch, parse error
```

## Run

```bash
npm install
npm run typecheck
npm test
npm start            # self-terminating fixture demo
npm run serve        # real stdio loop (waits on stdin)
```
