# Capstone 06 - DevOps Troubleshooting Agent (TypeScript)

الهيكل العظمي للتكامل البطيء للوكيل تحت الطلب في `../main.py`. يعرض أ
نقطة نهاية أمر الشرطة المائلة ونقطة نهاية التفاعل (النقر فوق الزر)، وكلاهما مسور
بواسطة Slack's HMAC-SHA256 طلب التوقيع بالإضافة إلى نافذة إعادة التشغيل مدتها 5 دقائق.
لا يتم تشغيل المعالجات المدمرة إلا بعد الموافقة على بطاقة Slack.

## Layout

```text
ts/
  package.json
  tsconfig.json
  src/
    index.ts          # entrypoint, demo + HTTP server
    server.ts         # hono app, /slack/command + /slack/interactivity
    slack_verify.ts   # HMAC v0 verification + timing-safe compare
    agent.ts          # mocked hypothesis ranker
    blocks.ts         # Block Kit response builder
    types.ts          # Hypothesis, AgentReport, SlackResponse, OutboundCall
  tests/
    slack_verify.test.ts
    agent.test.ts
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

قم بتعيين `SLACK_SIGNING_SECRET=...` لتجاوز سر العنصر النائب. ال
يقوم الخادم التفاعلي بطباعة المنفذ المختار (عشوائيًا عند عدم تعيين `PORT`).

## Tests

`node --test` عداء عبر tsx. التغطية:

- التحقق من التوقيع Slack: التوقيع صالح، والتوقيع تم التلاعب به تم رفضه، تم رفض الطابع الزمني الذي لا معنى له (> 5 دقائق انحراف)، ويتم رفض الطابع الزمني غير الرقمي مرفوض، يتم ممارسة مسار عدم تطابق الطول قبل مقارنة الوقت الثابت.
- الوكيل الوهمي: OOM مسار الكلمة الرئيسية، مسار الكلمة الرئيسية CrashLoop، المسار الاحتياطي.
- الخادم: `/health`، `/slack/command` مسارات سعيدة/معبثة/قديمة، `/slack/interactivity` الموافقة على الإجراء.
