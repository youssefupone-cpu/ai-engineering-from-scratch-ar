"""المرحلة 13 الدرس 22 - SKILL.md العرض التوضيحي لحزمة الوكيل والمحمل. يوزع ملفات SKILL.md باستخدام محلل stdlib YAML-frontmatter (بدون pyyaml)،
ينشئ سجل مهارات في الذاكرة، ويحاكي حلقة الوكيل التي يتم تحميلها
مهارة بالاسم وتستخدمها لبادئة موجه النظام. المهارات موجودة ضمن./skills/*/SKILL.md (تم إنشاؤها في /tmp لهذا العرض التوضيحي). تشغيل: كود بايثون/main.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SKILL_ROOT = Path("/tmp/lesson-21-skills")


# ------------------------------------------------------------------
# مهارات تركيب الألعاب
# ------------------------------------------------------------------

RELEASE_NOTES_SKILL = """\
---
الاسم: كاتب ملاحظات الإصدار
الوصف: اكتب إدخال سجل التغيير لأحدث العلاقات العامة المدمجة باتباع نمط هذا المشروع.
--- # كاتب ملاحظات الإصدار عند الاستدعاء، قم بتنفيذ الخطوات التالية: 1. قم بإدراج العلاقات العامة المدمجة منذ العلامة الأخيرة.
2. التجميع حسب التصنيف: الميزة، الإصلاح، العمل الرتيب، المستندات.
3. لكل PR، اكتب سطرًا واحدًا: `- <title> (#<num>)`.
4. قم بصياغة ملاحظات الإصدار وعرضها في CHANGELOG.md. إذا قال المستخدم "سفينة"، فقم بتشغيل `__TERM_0__ tag vX.Y.Z` و`gh release create`. راجع style-guide.md للتعرف على قواعد نمط المنزل.
"""

RELEASE_STYLE = """\
# دليل أسهل نسخة للنسخة - سطر واحد لكل PR. لا النثر.
- إدخالات الميزة أولاً؛ الإصلاحات الثانية؛ الأعمال المنزلية الثالثة؛ المستندات أخيرًا.
- تخطي الأعمال المنزلية من سجل التغيير العام.
"""

PR_REVIEW_SKILL = """\
---
الاسم: مراجع العلاقات العامة
الوصف: راجع الفرق PR مقابل دليل أسلوب المشروع وافتح التعليقات التوضيحية.
--- # PR مراجع الخطوات: 1. قم بإحضار الفرق PR.
2. حدد القواعد من AGENTS.md التي يمسها الفرق.
3. كتابة تعليق واحد لكل مخالفة واضحة.
"""


def setup_fixtures() -> None:
    SKILL_ROOT.mkdir(parents=True, exist_ok=True)
    rn = SKILL_ROOT / "release-notes-writer"
    rn.mkdir(exist_ok=True)
    (rn / "SKILL.md").write_text(RELEASE_NOTES_SKILL)
    (rn / "style-guide.md").write_text(RELEASE_STYLE)
    pr = SKILL_ROOT / "pr-reviewer"
    pr.mkdir(exist_ok=True)
    (pr / "SKILL.md").write_text(PR_REVIEW_SKILL)


# ------------------------------------------------------------------
# loader
# ------------------------------------------------------------------

@dataclass
class Skill:
    name: str
    description: str
    body: str
    root: Path


def parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text
    fm_raw = text[4:end]
    body = text[end + 5:]
    fm: dict = {}
    for line in fm_raw.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    return fm, body


def load_skill(folder: Path) -> Skill | None:
    skill_md = folder / "SKILL.md"
    if not skill_md.exists():
        return None
    text = skill_md.read_text()
    fm, body = parse_frontmatter(text)
    if "name" not in fm:
        return None
    return Skill(name=fm["name"], description=fm.get("description", ""),
                 body=body.strip(), root=folder)


def discover_skills(root: Path) -> dict[str, Skill]:
    registry: dict[str, Skill] = {}
    if not root.exists():
        return registry
    for item in sorted(root.iterdir()):
        if item.is_dir():
            s = load_skill(item)
            if s:
                registry[s.name] = s
    return registry


def read_subresource(skill: Skill, filename: str) -> str:
    path = skill.root / filename
    if not path.exists():
        return f"(no such subresource: {filename})"
    return path.read_text()


# ------------------------------------------------------------------
# حلقة الوكيل التجريبي
# ------------------------------------------------------------------

def agent_run(skill: Skill, user_task: str) -> str:
    print(f"  [loader] loading skill '{skill.name}'")
    print(f"  [loader] progressive disclosure: read style-guide only if needed")
    system_prompt = f"""أنت مساعد وقد تم تحميل مهارة {skill.name}. تعليمات المهارة:
{مهارة.الجسم} مهمة المستخدم: {user_task}
"""
    # إثبات الإفصاح التدريجي
    if "style-guide" in skill.body.lower():
        style = read_subresource(skill, "style-guide.md")
        print(f"  [loader] subresource pulled ({len(style)} bytes)")
        system_prompt += f"\n\nAdditional style guide:\n{style}"
    return system_prompt


def demo() -> None:
    print("=" * 72)
    print("PHASE 13 LESSON 21 - SKILLS AND AGENT SDK LOADER")
    print("=" * 72)

    setup_fixtures()

    print(f"\n--- discovery under {SKILL_ROOT} ---")
    skills = discover_skills(SKILL_ROOT)
    for name, s in skills.items():
        print(f"  {name:25s} -> {s.description}")

    print(f"\n--- invoke release-notes-writer with a fake user task ---")
    prompt = agent_run(skills["release-notes-writer"],
                       "draft the 1.4.0 release notes")
    print(f"\n[the system prompt the agent would send to the model]")
    print("-" * 72)
    print(prompt[:600] + "...")

    print("\n--- AGENTS.md + SKILL.md + MCP : the three-layer stack ---")
    print("  AGENTS.md (repo root)   -> project conventions at session start")
    print("  SKILL.md (./skills/*/)  -> reusable workflows on demand")
    print("  MCP server              -> tools the skill invokes (Phase 13 / 06-14)")


if __name__ == "__main__":
    demo()
