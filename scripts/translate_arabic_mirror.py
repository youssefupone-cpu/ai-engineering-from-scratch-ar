#!/usr/bin/env python3
"""
Build an Arabic mirror of the curriculum repository.

What this script does:
- Mirrors the source tree into /home/youssef-ayad/ai-engineering-from-scratch-ar
- Translates Markdown, JSON, HTML, SVG, and JS content that is user-facing
- Preserves code, URLs, paths, and structural keys that the site depends on
- Prunes stale files from the output tree so it matches the source tree
"""

from __future__ import annotations

import html as html_lib
import json
import os
import re
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from lxml import etree, html as lxml_html


SOURCE_ROOT = Path(os.environ.get("SOURCE_ROOT", "/home/youssef-ayad/ai-engineering-from-scratch"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/home/youssef-ayad/ai-engineering-from-scratch-ar"))

EXCLUDE_DIRS = {
    ".git",
    ".codex",
    ".agents",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "node_modules",
    ".venv",
}

TEXT_FILE_EXTS = {
    ".md",
    ".markdown",
    ".mkd",
    ".json",
    ".html",
    ".htm",
    ".svg",
    ".js",
    ".mjs",
    ".cjs",
    ".sh",
}

GOOGLE_TRANSLATE_URL = "https://translate.googleapis.com/translate_a/single"
MAX_CHUNK_CHARS = 3500

# Keep product names and technical tokens stable.
PROTECTED_TERMS = [
    "uv",
    "pip",
    "npm",
    "git",
    "curl",
    "wget",
    "bash",
    "sudo",
    "ssh",
    "docker",
    "make",
    "pytest",
    "conda",
    "venv",
    "virtualenv",
    "juliaup",
    "fnm",
    "node",
    "pnpm",
    "cargo",
    "Rust",
    "TypeScript",
    "Node.js",
    "PyTorch",
    "Jupyter",
    "GitHub",
    "Ollama",
    "Google Colab",
    "NumPy",
    "scikit-learn",
    "OpenAI",
    "Hugging Face",
    "MCP",
    "LLM",
    "LLMs",
    "RAG",
    "API",
    "APIs",
    "CLI",
    "CPU",
    "GPU",
    "CSS",
    "HTML",
    "JSON",
    "SVG",
    "PNG",
    "SSE",
    "WebSocket",
    "MOTA",
    "IDF1",
    "HOTA",
    "CLIP",
    "LoRA",
    "RLHF",
    "SFT",
    "BPE",
    "WordPiece",
    "TF-IDF",
]

PLACEHOLDER_MARKS = {
    "TERM": ("⟦", "⟧"),
    "URL": ("⟪", "⟫"),
    "LINK": ("⟨", "⟩"),
    "CODE": ("‹", "›"),
    "TAG": ("❬", "❭"),
    "ESC": ("⌈", "⌉"),
}

SHORT_REPLACEMENTS = {
    "Skip to content": "تخطَّ إلى المحتوى",
    "Contents": "المحتويات",
    "Catalog": "الفهرس",
    "Roadmap": "خارطة الطريق",
    "Glossary": "المعجم",
    "Home": "الرئيسية",
    "Search": "بحث",
    "Search (⌘K)": "بحث (⌘K)",
    "Search lessons...": "ابحث عن الدروس...",
    "Search terms...": "ابحث في المصطلحات...",
    "Toggle theme": "تبديل السمة",
    "Complete": "مكتمل",
    "Planned": "مخطط",
    "In progress": "قيد التنفيذ",
    "Read": "اقرأ",
    "Review": "راجع",
    "Test Your Understanding": "اختبر فهمك",
    "Did you get it?": "هل استوعبت الفكرة؟",
    "Complete all questions to see your score": "أجب عن جميع الأسئلة لرؤية نتيجتك",
    "Perfect score!": "نتيجة كاملة!",
    "Great work!": "عمل رائع!",
    "Keep studying!": "تابع الدراسة!",
    "Learning Path": "مسار التعلم",
    "Continue Learning": "تابع التعلم",
    "full course catalog": "فهرس الدورة الكامل",
    "Want a deeper quiz? Run": "هل تريد اختبارًا أعمق؟ شغّل",
    "You completed this lesson": "أكملت هذا الدرس",
    "Mark as not done": "تمييز كغير مكتمل",
    "Mark complete": "تمييز كمكتمل",
    "completed": "مكتمل",
    "Question": "سؤال",
    "Score": "النتيجة",
    "Phase": "المرحلة",
}

PY_COMMENT_SKIP_PREFIXES = ("requires:", "type:", "path:", "phase:")

JSON_SKIP_KEYS = {
    "status",
    "type",
    "lang",
    "url",
    "path",
    "lessonPath",
    "id",
    "kind",
    "artKind",
    "combines",
    "phase",
    "lesson",
    "slug",
    "version",
    "schema_version",
    "file",
}

JSON_TRANSLATE_KEYS = {
    "name",
    "title",
    "desc",
    "description",
    "summary",
    "keywords",
    "term",
    "says",
    "means",
    "question",
    "explanation",
    "prompt",
    "message",
    "body",
    "text",
    "content",
    "label",
    "caption",
}

HTML_TEXT_TAGS = {"title", "p", "li", "th", "td", "span", "a", "button", "h1", "h2", "h3", "h4", "h5", "h6", "option", "legend", "label", "strong", "em", "small", "figcaption", "summary", "code"}
HTML_ATTRS = {"title", "aria-label", "placeholder", "alt", "content"}

PROTECTED_TERM_RE = re.compile(
    "|".join(sorted((re.escape(term) for term in PROTECTED_TERMS), key=len, reverse=True))
)
PROTECTED_ACRONYM_RE = re.compile(r"\b[A-Z][A-Z0-9]{1,6}\b")
URL_RE = re.compile(r"https?://[^\s<>\"]+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
HTML_TAG_RE = re.compile(r"<[^>]+>")
ESCAPE_RE = re.compile(r"\\.")
SHELL_HEREDOC_RE = re.compile(r"<<['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
JS_STRING_RE = re.compile(r'(?<!\\)(["\'])(.*?)(?<!\\)\1', re.S)
PY_TRIPLE_RE = re.compile(r'([rRuUfF]{0,2})(\"\"\"|\'\'\')([\s\S]*?)(\2)', re.S)


def is_probably_binary(path: Path) -> bool:
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return True
    if b"\0" in chunk:
        return True
    try:
        chunk.decode("utf-8")
        return False
    except UnicodeDecodeError:
        return True


def split_long_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for segment in re.split(r"(\s+)", text):
        if not segment:
            continue
        if current and current_len + len(segment) > max_chars:
            chunks.append("".join(current))
            current = [segment]
            current_len = len(segment)
        else:
            current.append(segment)
            current_len += len(segment)
    if current:
        chunks.append("".join(current))
    return chunks


def protect_terms(text: str) -> tuple[str, dict[str, str]]:
    placeholders: dict[str, str] = {}
    counter = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        left, right = PLACEHOLDER_MARKS["TERM"]
        placeholder = f"{left}{counter}{right}"
        placeholders[placeholder] = match.group(0)
        counter += 1
        return placeholder

    text = PROTECTED_TERM_RE.sub(replace, text)
    text = PROTECTED_ACRONYM_RE.sub(replace, text)
    return text, placeholders


def restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    for placeholder, value in placeholders.items():
        text = text.replace(placeholder, value)
    return text


def normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?؟،])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def looks_code_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped.startswith(("http://", "https://", "data:", "phases/", "site/", "scripts/")):
        return True
    if stripped.startswith((".", "#", "/", "<")):
        return True
    if re.fullmatch(r"[A-Za-z0-9_./\\:-]+", stripped):
        return True
    if re.fullmatch(r"[A-Z0-9_]+", stripped):
        return True
    if stripped.upper() == stripped and "/" in stripped:
        return True
    return False


def should_translate(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if stripped in SHORT_REPLACEMENTS:
        return True
    if len(stripped) < 4:
        return False
    if looks_code_like(stripped):
        return False
    if not re.search(r"[A-Za-z]", stripped):
        return False
    if re.search(r"\s", stripped):
        return True
    return False


@lru_cache(maxsize=32768)
def translate_via_google(text: str, source_language: str = "en", target_language: str = "ar") -> str:
    if not text or not text.strip():
        return text
    params = {
        "client": "gtx",
        "sl": source_language,
        "tl": target_language,
        "dt": "t",
        "q": text,
    }
    url = f"{GOOGLE_TRANSLATE_URL}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            parts = []
            for segment in payload[0]:
                if segment and segment[0]:
                    parts.append(segment[0])
            return html_lib.unescape("".join(parts))
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = exc
            time.sleep(0.4 * (attempt + 1))
    print(f"translation failed, keeping source text: {last_error}")
    return text


def translate_fragment(text: str, preserve_html: bool = False) -> str:
    if not should_translate(text):
        return text

    lead = re.match(r"^\s*", text).group(0) if text else ""
    trail = re.search(r"\s*$", text).group(0) if text else ""
    core = text[len(lead) : len(text) - len(trail) if trail else len(text)]
    if not core:
        return text
    if core in SHORT_REPLACEMENTS:
        return lead + SHORT_REPLACEMENTS[core] + trail

    temp = core
    temp = temp.replace("\\n", "__NL__").replace("\\t", "__TAB__")
    temp, term_placeholders = protect_terms(temp)
    url_placeholders: dict[str, str] = {}

    def protect_regex(pattern: re.Pattern[str], prefix: str, value: str) -> str:
        left, right = PLACEHOLDER_MARKS[prefix]
        placeholder = f"{left}{len(url_placeholders)}{right}"
        url_placeholders[placeholder] = value
        return placeholder

    temp = URL_RE.sub(lambda m: protect_regex(URL_RE, "URL", m.group(0)), temp)
    temp = MD_LINK_RE.sub(lambda m: protect_regex(MD_LINK_RE, "LINK", m.group(0)), temp)
    temp = INLINE_CODE_RE.sub(lambda m: protect_regex(INLINE_CODE_RE, "CODE", m.group(0)), temp)
    if preserve_html:
        temp = HTML_TAG_RE.sub(lambda m: protect_regex(HTML_TAG_RE, "TAG", m.group(0)), temp)

    chunks = [translate_via_google(chunk) for chunk in split_long_text(temp)]
    translated = "".join(chunks)
    translated = restore_placeholders(translated, term_placeholders)
    for placeholder, value in url_placeholders.items():
        translated = translated.replace(placeholder, value)
    translated = translated.replace("__NL__", "\\n").replace("__TAB__", "\\t")
    translated = normalize_spacing(translated)
    return lead + translated + trail


def translate_markdown(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_code = False
    in_frontmatter = False
    seen_content = False
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal buffer
        if not buffer:
            return
        chunk = "".join(buffer)
        if chunk.strip():
            out.append(translate_fragment(chunk, preserve_html=False))
        else:
            out.append(chunk)
        buffer = []

    for line in lines:
        stripped = line.strip()
        if not seen_content and stripped == "---":
            flush_buffer()
            in_frontmatter = not in_frontmatter
            out.append(line)
            continue
        if in_frontmatter:
            out.append(line)
            continue
        if stripped.startswith("```") or stripped.startswith("~~~"):
            flush_buffer()
            in_code = not in_code
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        if stripped in ("---", "***"):
            flush_buffer()
            out.append(line)
            continue
        if not stripped:
            flush_buffer()
            out.append(line)
            continue
        seen_content = True
        buffer.append(line)
    flush_buffer()
    return "".join(out)


def translate_json_value(key: str | None, value: Any) -> Any:
    if isinstance(value, dict):
        return {k: translate_json_value(k, v) for k, v in value.items()}
    if isinstance(value, list):
        if key == "tags":
            return value
        return [translate_json_value(key, item) for item in value]
    if isinstance(value, str):
        if key in JSON_SKIP_KEYS:
            return value
        if key in JSON_TRANSLATE_KEYS or should_translate(value):
            return translate_fragment(value, preserve_html=False)
        return value
    return value


def translate_json(text: str) -> str:
    data = json.loads(text)
    translated = translate_json_value(None, data)
    return json.dumps(translated, ensure_ascii=False, indent=2) + "\n"


def translate_site_data_js(text: str) -> str:
    def extract_block(name: str) -> tuple[str, int, int]:
        marker = f"const {name} ="
        start = text.index(marker)
        bracket_start = text.index("[", start)
        depth = 0
        in_str: str | None = None
        escape = False
        for i in range(bracket_start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == in_str:
                    in_str = None
                continue
            if ch in ("'", '"'):
                in_str = ch
                continue
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    while end < len(text) and text[end].isspace():
                        end += 1
                    if end < len(text) and text[end] == ";":
                        return text[bracket_start:end], bracket_start, end + 1
        raise ValueError(f"could not locate array for {name}")

    def recurse(value: Any, key: str | None = None) -> Any:
        if isinstance(value, dict):
            return {k: recurse(v, k) for k, v in value.items()}
        if isinstance(value, list):
            if key == "tags":
                return value
            return [recurse(item, key) for item in value]
        if isinstance(value, str):
            if key in JSON_SKIP_KEYS:
                return value
            if key in JSON_TRANSLATE_KEYS or should_translate(value):
                return translate_fragment(value, preserve_html=False)
            return value
        return value

    out = text
    blocks: list[tuple[int, int, str, str]] = []
    for name in ("PHASES", "GLOSSARY", "ARTIFACTS"):
        block, start, end = extract_block(name)
        blocks.append((start, end, name, block))
    for start, end, name, block in sorted(blocks, reverse=True):
        data = json.loads(block)
        translated = recurse(data)
        replacement = json.dumps(translated, ensure_ascii=False, indent=2)
        out = out[:start] + replacement + out[end:]
    return out


def translate_html(text: str, source_path: Path) -> str:
    parser = lxml_html.HTMLParser(encoding="utf-8")
    root = lxml_html.fromstring(text, parser=parser)

    if isinstance(root.tag, str) and root.tag.lower() == "html":
        root.set("lang", "ar")

    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = etree.QName(el).localname.lower()
        if tag == "script":
            if el.text and el.text.strip():
                el.text = translate_js_source(el.text)
            continue
        if tag == "style":
            continue
        if el.text and el.text.strip() and should_translate(el.text):
            el.text = translate_fragment(el.text, preserve_html=False)
        for child in el:
            if child.tail and child.tail.strip() and should_translate(child.tail):
                child.tail = translate_fragment(child.tail, preserve_html=False)
        for attr in list(el.attrib):
            if attr in HTML_ATTRS:
                value = el.attrib[attr]
                if value and should_translate(value):
                    el.attrib[attr] = translate_fragment(value, preserve_html=False)

    return lxml_html.tostring(root, encoding="unicode", method="html")


def translate_svg(text: str) -> str:
    parser = etree.XMLParser(remove_blank_text=False, recover=True)
    root = etree.fromstring(text.encode("utf-8"), parser=parser)
    for el in root.iter():
        if not isinstance(el.tag, str):
            continue
        tag = etree.QName(el).localname.lower()
        if tag in {"text", "tspan", "title", "desc"} and el.text and el.text.strip():
            el.text = translate_fragment(el.text, preserve_html=False)
        for attr in ("aria-label", "title"):
            if attr in el.attrib and should_translate(el.attrib[attr]):
                el.attrib[attr] = translate_fragment(el.attrib[attr], preserve_html=False)
    return etree.tostring(root, encoding="unicode")


def read_js_string(text: str, start: int, quote: str) -> tuple[str, int]:
    body: list[str] = []
    i = start + 1
    escaped = False
    while i < len(text):
        ch = text[i]
        if escaped:
            body.append(ch)
            escaped = False
        elif ch == "\\":
            body.append(ch)
            escaped = True
        elif ch == quote:
            return "".join(body), i
        else:
            body.append(ch)
        i += 1
    return "".join(body), len(text) - 1


def translate_js_string_content(raw: str) -> str:
    stripped = raw.strip()
    if stripped in SHORT_REPLACEMENTS:
        translated = SHORT_REPLACEMENTS[stripped]
        return raw.replace(stripped, translated, 1)
    if not should_translate(raw):
        return raw
    temp = raw
    temp, term_placeholders = protect_terms(temp)
    placeholders: dict[str, str] = {}

    def hold(value: str, prefix: str) -> str:
        left, right = PLACEHOLDER_MARKS[prefix]
        placeholder = f"{left}{len(placeholders)}{right}"
        placeholders[placeholder] = value
        return placeholder

    temp = URL_RE.sub(lambda m: hold(m.group(0), "URL"), temp)
    temp = HTML_TAG_RE.sub(lambda m: hold(m.group(0), "TAG"), temp)
    temp = ESCAPE_RE.sub(lambda m: hold(m.group(0), "ESC"), temp)
    chunks = [translate_via_google(chunk) for chunk in split_long_text(temp)]
    translated = "".join(chunks)
    translated = restore_placeholders(translated, term_placeholders)
    for placeholder, value in placeholders.items():
        translated = translated.replace(placeholder, value)
    translated = normalize_spacing(translated)
    return translated


def translate_js_source(text: str) -> str:
    if not text:
        return text
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = i + 2
            while j < n and text[j] not in "\r\n":
                j += 1
            comment = text[i + 2 : j]
            translated = translate_fragment(comment, preserve_html=False) if should_translate(comment) else comment
            out.append("//" + translated)
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                j = n - 2
            comment = text[i + 2 : j]
            translated = translate_fragment(comment, preserve_html=False) if should_translate(comment) else comment
            out.append("/*" + translated + "*/")
            i = j + 2
            continue
        if ch in ("'", '"'):
            raw, end = read_js_string(text, i, ch)
            out.append(ch + translate_js_string_content(raw) + ch)
            i = end + 1
            continue
        if ch == "`":
            raw, end = read_js_string(text, i, "`")
            if "${" not in raw and should_translate(raw):
                out.append("`" + translate_js_string_content(raw) + "`")
            else:
                out.append("`" + raw + "`")
            i = end + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def translate_python_source(text: str) -> str:
    def replace_docstring(match: re.Match[str]) -> str:
        prefix, quote, body, _ = match.groups()
        if not should_translate(body):
            return match.group(0)
        return f"{prefix}{quote}{translate_fragment(body, preserve_html=False)}{quote}"

    def replace_comment(match: re.Match[str]) -> str:
        prefix, body = match.groups()
        stripped = body.strip()
        if not stripped or stripped.startswith(PY_COMMENT_SKIP_PREFIXES):
            return match.group(0)
        if not should_translate(stripped):
            return match.group(0)
        return prefix + translate_fragment(stripped, preserve_html=False)

    text = PY_TRIPLE_RE.sub(replace_docstring, text)
    text = re.sub(r"(?m)^(\s*#\s?)(.*)$", replace_comment, text)
    return text


def translate_shell_source(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_heredoc: str | None = None
    collecting: list[str] = []
    heredoc_indent = ""

    def flush_heredoc() -> None:
        nonlocal collecting
        if not collecting:
            return
        body = "".join(collecting)
        translated_lines = []
        for raw_line in body.splitlines(keepends=True):
            stripped = raw_line.strip()
            if not stripped:
                translated_lines.append(raw_line)
                continue
            if stripped.startswith(("scripts/", "python3 ", "python ", "node ", "bash ", "sh ", "uv ", "git ")):
                translated_lines.append(raw_line)
                continue
            translated_lines.append(translate_fragment(raw_line, preserve_html=False))
        out.extend(translated_lines)
        collecting = []

    for line in lines:
        stripped = line.strip()
        if in_heredoc:
            if stripped == in_heredoc:
                flush_heredoc()
                out.append(line)
                in_heredoc = None
                continue
            collecting.append(line)
            continue
        if line.startswith("#!") or not stripped:
            out.append(line)
            continue
        if stripped.startswith("#"):
            comment = stripped[1:].lstrip()
            if comment and not comment.startswith(PY_COMMENT_SKIP_PREFIXES):
                translated = translate_fragment(comment, preserve_html=False) if should_translate(comment) else comment
                prefix = line[: line.index("#")]
                out.append(prefix + "# " + translated + ("\n" if line.endswith("\n") else ""))
            else:
                out.append(line)
            continue
        heredoc_match = SHELL_HEREDOC_RE.search(line)
        if heredoc_match:
            in_heredoc = heredoc_match.group(1)
            out.append(line)
            continue
        echo_match = re.match(r'^(?P<indent>\s*echo\s+)(["\'])(?P<body>.*)(["\'])\s*$', line)
        if echo_match:
            body = echo_match.group("body")
            if should_translate(body):
                translated = translate_fragment(body, preserve_html=False)
                out.append(f"{echo_match.group('indent')}{echo_match.group(2)}{translated}{echo_match.group(4)}")
            else:
                out.append(line)
            continue
        out.append(line)

    if collecting:
        flush_heredoc()
    return "".join(out)


def translate_text_file(path: Path, text: str) -> str:
    suffix = path.suffix.lower()
    if path.name == "data.js" and path.parent.name == "site":
        return translate_site_data_js(text)
    if suffix in {".md", ".markdown", ".mkd"}:
        return translate_markdown(text)
    if suffix == ".json":
        return translate_json(text)
    if suffix in {".html", ".htm"}:
        return translate_html(text, path)
    if suffix == ".svg":
        return translate_svg(text)
    if suffix in {".js", ".mjs", ".cjs", ".ts", ".tsx"}:
        return translate_js_source(text)
    if suffix == ".py":
        return translate_python_source(text)
    if suffix == ".sh":
        rel = path.relative_to(SOURCE_ROOT).as_posix()
        if rel == "scripts/scaffold-lesson.sh":
            return translate_shell_source(text)
        return text
    return text


def mirror_one(src_path: Path, dst_path: Path) -> tuple[str, str]:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    rel = src_path.relative_to(SOURCE_ROOT).as_posix()

    if src_path.suffix.lower() in {".py", ".ts", ".tsx"}:
        shutil.copy2(src_path, dst_path)
        return "copy", rel

    if src_path.suffix.lower() in {".js", ".mjs", ".cjs"} and not rel.startswith("site/"):
        shutil.copy2(src_path, dst_path)
        return "copy", rel

    if src_path.suffix.lower() == ".sh" and rel != "scripts/scaffold-lesson.sh":
        shutil.copy2(src_path, dst_path)
        return "copy", rel

    if is_probably_binary(src_path) or src_path.suffix.lower() not in TEXT_FILE_EXTS:
        shutil.copy2(src_path, dst_path)
        return "copy", rel

    try:
        raw = src_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        shutil.copy2(src_path, dst_path)
        return "copy", rel

    translated = translate_text_file(src_path, raw)
    if translated == raw and dst_path.exists():
        # Preserve identical files without rewriting if nothing changed.
        return "skip", rel

    dst_path.write_text(translated, encoding="utf-8")
    shutil.copymode(src_path, dst_path)
    return "translate", src_path.relative_to(SOURCE_ROOT).as_posix()


def iter_source_files() -> list[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(SOURCE_ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        rel_dir = Path(dirpath).relative_to(SOURCE_ROOT)
        if rel_dir.parts and rel_dir.parts[0] in EXCLUDE_DIRS:
            continue
        for name in filenames:
            src = Path(dirpath) / name
            if OUTPUT_ROOT in src.parents:
                continue
            files.append(src)
    files.sort(key=lambda p: p.relative_to(SOURCE_ROOT).as_posix())
    return files


def prune_extras(expected: set[str]) -> int:
    removed = 0
    if not OUTPUT_ROOT.exists():
        return 0
    for dirpath, dirnames, filenames in os.walk(OUTPUT_ROOT, topdown=False):
        for filename in filenames:
            path = Path(dirpath) / filename
            rel = path.relative_to(OUTPUT_ROOT).as_posix()
            if rel not in expected:
                try:
                    path.unlink()
                    removed += 1
                except FileNotFoundError:
                    pass
        for dirname in dirnames:
            path = Path(dirpath) / dirname
            try:
                if not any(path.iterdir()):
                    path.rmdir()
            except FileNotFoundError:
                pass
    return removed


def main() -> int:
    force = os.environ.get("FORCE_TRANSLATE", "").strip().lower() in {"1", "true", "yes"}
    max_workers = int(os.environ.get("TRANSLATE_WORKERS", "4"))

    files = iter_source_files()
    expected = {p.relative_to(SOURCE_ROOT).as_posix() for p in files}
    stats = {"translate": 0, "copy": 0, "skip": 0, "failed": 0}

    tasks: list[tuple[Path, Path]] = []
    for src in files:
        dst = OUTPUT_ROOT / src.relative_to(SOURCE_ROOT)
        if not force and dst.exists() and dst.is_file() and dst.stat().st_size > 0:
            # Re-run translation only when the source is newer or the file is missing.
            if src.stat().st_mtime <= dst.stat().st_mtime:
                stats["skip"] += 1
                continue
        tasks.append((src, dst))

    if tasks:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(mirror_one, src, dst): (src, dst) for src, dst in tasks}
            for future in as_completed(future_map):
                src, _ = future_map[future]
                rel = src.relative_to(SOURCE_ROOT).as_posix()
                try:
                    action, _ = future.result()
                    stats[action] += 1
                except Exception as exc:  # pragma: no cover - keep going on isolated failures
                    stats["failed"] += 1
                    print(f"error: {rel}: {exc}")

    removed = prune_extras(expected)

    print("\nArabic mirror complete.")
    print(f"  translated: {stats['translate']}")
    print(f"  copied: {stats['copy']}")
    print(f"  skipped: {stats['skip']}")
    print(f"  failed: {stats['failed']}")
    print(f"  pruned extra files: {removed}")
    print(f"  output: {OUTPUT_ROOT}")
    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
