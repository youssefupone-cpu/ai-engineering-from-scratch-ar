# Skills and Agent SDKs — Anthropic Skills, AGENTS.md, OpenAI Apps SDK

> MCP يقول "ما الأدوات الموجودة؟" المهارات تقول "كيفية القيام بالمهمة". طبقات المكدس 2026 كلاهما. يتم شحن مهارات وكيل Anthropic (المعيار المفتوح، ديسمبر 2025) على أنها SKILL.md مع الكشف التدريجي. تطبيقات OpenAI SDK هي MCP بالإضافة إلى البيانات التعريفية للقطعة. AGENTS.md (الآن في أكثر من 60.000+ إعادة شراء) يقع في جذر الريبو كسياق وكيل على مستوى المشروع. يسمي هذا الدرس ما يغطيه كل منهم ويبني الحد الأدنى من حزمة SKILL.md + AGENTS.md التي تنتقل عبر الوكلاء.

**النوع:** تعلم
** اللغات: ** بايثون (stdlib، SKILL.md محلل ومحمل)
**المتطلبات:** المرحلة 13 · 07 (MCP خادم)
**الوقت:** ~45 دقيقة

## Learning Objectives

- التمييز بين الطبقات الثلاث: AGENTS.md (سياق المشروع)، SKILL.md (المعرفة القابلة لإعادة الاستخدام)، MCP (الأدوات).
- اكتب SKILL.md مع YAML المادة الأمامية والكشف التدريجي.
- تحميل نمط نظام ملفات المهارات في وقت تشغيل الوكيل.
- قم بتكوين مهارة باستخدام خادم MCP وAGENTS.md بحيث تعمل حزمة واحدة في Claude Code وCursor وCodex.

## The Problem

يقوم أحد المهندسين بتقطير سير عمل كتابة ملاحظات الإصدار في موجه متعدد الخطوات: "اقرأ أحدث العلاقات العامة المدمجة. مجموعة حسب المنطقة. قم بتلخيص كل منها. اكتب إدخال سجل التغيير وفقًا لأسلوب الفريق. انشر على مسودة Slack." لقد وضعوها في مستند Notion لفريقهم.

يريدون الآن استخدام سير العمل هذا من Claude Code وCursor وCodex CLI. كل وكيل لديه طريقة مختلفة لتحميل التعليمات: أوامر الشرطة المائلة لـ Claude Code، قواعد المؤشر، Codex `.codex.md`. يقوم المهندس بنسخ سير العمل ثلاث مرات ويحتفظ بثلاث نسخ.

AGENTS.md و SKILL.md معًا يصلحان هذا:

- **AGENTS.md** يقع في جذر الريبو. يقرأها كل وكيل متوافق عند بدء الجلسة. "كيف يعمل هذا المشروع؟ ما هي الاتفاقيات؟ ما هي الأوامر التي تجري الاختبارات؟"
- **SKILL.md** عبارة عن حزمة محمولة: YAML المادة الأمامية (الاسم والوصف) + نص تخفيض السعر + موارد اختيارية. يقوم الوكلاء الذين يدعمون المهارات بتحميلها بالاسم عند الطلب.
- **MCP** (المرحلة 13 · 06-14) تتعامل مع الأدوات التي تحتاج المهارة إلى استدعائها.

ثلاث طبقات، قطعة أثرية واحدة محمولة.

## The Concept

### AGENTS.md (agents.md)

تم إطلاقه في أواخر عام 2025، وتبعه أكثر من 60,000 إعادة شراء بحلول أبريل 2026. ملف واحد في جذر الريبو. شكل:

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

يقرأ الوكلاء هذا عند بدء الجلسة ويستخدمونه لمعايرة سلوكهم لهذا المشروع. يدعم كل وكيل ترميز في عام 2026 AGENTS.md: Claude Code، Cursor، Codex، Copilot Workspace، opencode، Windsurf، Zed.

### SKILL.md format

مهارات الوكيل الأنثروبي (تم إصدارها كمعيار مفتوح في ديسمبر 2025):

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

تعلن Frontmatter عن هوية المهارة. الجسم هو الموجه الذي يظهر للنموذج عند تحميل المهارة.

### Progressive disclosure

يمكن أن تشير المهارات إلى الموارد الفرعية التي يجلبها الوكيل فقط عند الحاجة. مثال:

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md يقول "راجع style-guide.md للتعرف على قواعد الأسلوب." يقوم الوكيل بسحب style-guide.md فقط عندما تكون المهارة قيد التشغيل بشكل نشط. يؤدي هذا إلى تجنب تضخيم الموجه بالتفاصيل التي قد لا يحتاجها النموذج.

### Filesystem discovery

تقوم أوقات تشغيل الوكيل بفحص الدلائل المعروفة بحثًا عن ملفات SKILL.md:

- `~/.anthropic/skills/*/SKILL.md`
- المشروع `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

يتم التحميل حسب اسم المجلد والجزء الأمامي `name`. يتبع كل من Claude Code وAnthropic Claude Agent SDK وSkillKit (الوكيل المشترك) هذا النمط.

### Anthropic Claude Agent SDK

يقوم `@anthropic-ai/claude-agent-sdk` (TypeScript) و `claude-agent-sdk` (Python) بتحميل المهارات عند بداية الجلسة، ويعرضونها كـ "وكلاء" قابلين للاستدعاء داخل وقت التشغيل. ترسل حلقة الوكيل إلى المهارة عندما يستدعيها المستخدم.

### OpenAI Apps SDK

تم إطلاقه في أكتوبر 2025؛ بنيت مباشرة على MCP. يوحد موصلات OpenAI السابقة وإجراءات GPT المخصصة تحت سطح مطور واحد. تطبيق Apps SDK هو:

- خادم MCP (أدوات، موارد، مطالبات).
- بالإضافة إلى البيانات التعريفية لعنصر واجهة المستخدم لـ ChatGPT UI.
- بالإضافة إلى مورد MCP Apps `ui://` اختياري للأسطح التفاعلية.

نفس البروتوكول أغنى UX.

### Cross-agent portability via SkillKit

تقوم أدوات مثل SkillKit وطبقات التوزيع المشابهة عبر الوكلاء بترجمة SKILL.md واحد إلى التنسيق الأصلي لكل من 32+ AI وكيل (Claude Code، Cursor، Codex، Gemini CLI، OpenCode، وما إلى ذلك). مصدر واحد للحقيقة؛ العديد من المستهلكين.

### The three-layer stack

| طبقة | ملف | تم التحميل عندما | الغرض |
|-------|------|------------|---------|
| AGENTS.md | جذر الريبو | بداية الجلسة | اتفاقيات على مستوى المشروع |
| SKILL.md | دليل المهارات | تم استدعاء المهارة | سير العمل القابل لإعادة الاستخدام |
| MCP الخادم | عملية خارجية | الأدوات اللازمة | إجراءات قابلة للاستدعاء |

يؤلف الثلاثة جميعًا: يقرأ الوكيل AGENTS.md عند بدء الجلسة، ويستدعي المستخدم مهارة، وتتضمن تعليمات المهارة MCP استدعاءات الأداة، ويرسل الوكيل عبر عميل MCP.

## Use It

`code/main.py` يشحن محلل ومحمل stdlib SKILL.md. يكتشف المهارات تحت `./skills/`، ويوزع YAML المادة الأمامية بالإضافة إلى جسم تخفيض السعر، وينتج إملاءًا مرتبطًا باسم المهارة. ثم يقوم بمحاكاة حلقة الوكيل التي تستدعي `release-notes-writer` بالاسم.

ما الذي يجب النظر إليه:

- YAML تم تحليل المادة الأمامية باستخدام الحد الأدنى من المحلل اللغوي stdlib (بدون تبعية `pyyaml`).
- هيئة المهارة المخزنة حرفيا. يقوم الوكيل بإضافته مسبقًا إلى موجه النظام عند الاستدعاء.
- تم عرض الكشف التدريجي عبر وظيفة `read_subresource` التي تسحب الملفات المرجعية عند الطلب.

## Ship It

ينتج عن هذا الدرس `outputs/skill-agent-bundle.md`. بالنظر إلى سير العمل، تنتج المهارة حزمة مخططات الخادم SKILL.md + AGENTS.md + MCP المدمجة، والتي يمكن نقلها عبر الوكلاء.

## Exercises

1. قم بتشغيل `code/main.py`. أضف مهارة ثانية ضمن `skills/` وتأكد من أن المُحمل يلتقطها.

2. اكتب AGENTS.md لهذه الدورة التدريبية. قم بتضمين أوامر الاختبار واصطلاحات الأسلوب والنموذج العقلي للمرحلة 13.

3. قم بنقل سير عمل متعدد الخطوات من المستندات الداخلية لفريقك إلى SKILL.md. تحقق من أنه يتم تحميله في Claude Code.

4. قم بترجمة المهارة إلى تنسيقات القواعد الأصلية الخاصة بـ Cursor و Codex يدويًا. قم بإحصاء الاختلافات بين التنسيقات - هذا هو سطح الترجمة الذي تقوم SkillKit بأتمتةه.

5. اقرأ منشور مدونة مهارات الوكيل الأنثروبي. حدد ميزة واحدة في وكيل كلود SDK لا يغطيها محمل هذا الدرس. (تلميح: استدعاء الوكيل الفرعي.)

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|-----------------------|
| SKILL.md | "ملف المهارة" | YAML المادة الأمامية بالإضافة إلى جسم تخفيض السعر، محملة بواسطة وقت تشغيل الوكيل |
| AGENTS.md | "سياق وكيل الريبو الجذر" | قراءة ملف الاتفاقيات على مستوى المشروع عند بدء الجلسة |
| الكشف التدريجي | "الموارد الفرعية ذات التحميل البطيء" | يتم سحب الملفات المرجعية لجسم المهارة فقط عند الحاجة إليها |
| فرونت ماتر | "YAML كتلة في الأعلى" | البيانات الوصفية (الاسم والوصف) في المحددات `---` |
| كلود الوكيل SDK | "وقت تشغيل المهارة الإنسانية" | `@anthropic-ai/claude-agent-sdk`، يحمل المهارات والطرق |
| OpenAI التطبيقات SDK | "MCP + تعريف القطعة" | سطح تطوير OpenAI مبني على خطافات MCP بالإضافة إلى ChatGPT UI |
| اكتشاف المهارات | "فحص نظام الملفات" | المشي معروف dirs لـ SKILL.md، المفتاح بالاسم |
| قابلية النقل عبر الوكيل | "مهارة واحدة العديد من العملاء" | ترجمة واحد SKILL.md إلى أكثر من 32 وكيلًا عبر أدوات بأسلوب SkillKit |
| مهارة الوكيل | "الدراية المحمولة" | قالب مهمة قابل لإعادة الاستخدام خارج مفهوم أداة MCP |
| التطبيقات SDK | "MCP بلس ChatGPT UI" | تم توحيد الموصلات ونقاط GPT المخصصة على MCP |

## Further Reading

- [إعلان مهارات الوكيل الأنثروبي](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) - الإطلاق في ديسمبر 2025
- [المستندات الأنثروبولوجية - مهارات الوكيل](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md مرجع التنسيق
- [OpenAI — التطبيقات SDK](https://developers.openai.com/apps-sdk) — MCP منصة المطورين لـ ChatGPT
- [agents.md](https://agents.md/) — AGENTS.md التنسيق وقائمة الاعتماد
- [الأنثروبولوجية - الأنثروبولوجية / المهارات GitHub](https://githubhub.com/anthropics/skills) - أمثلة المهارات الرسمية
