#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
الاستخدام: scripts/scaffold-lesson.sh <phase-dir> <lesson-slug> [العنوان]

Examples:
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers "Tokenizers from Scratch"

إنشاء مراحل/<phase-dir>/<lesson-slug>/ باستخدام الكود/، دفتر الملاحظات/، المستندات/، المخرجات/
وهيكل docs/en.md تمت تعبئته مسبقًا من LESSON_TEMPLATE.md.
USAGE
  exit 2
fi

PHASE="$1"
LESSON="$2"
TITLE="${3:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "error: run this from inside the ai-engineering-from-scratch git repo" >&2
  exit 1
fi

PHASE_DIR="$REPO_ROOT/phases/$PHASE"
LESSON_DIR="$PHASE_DIR/$LESSON"

if [[ ! -d "$PHASE_DIR" ]]; then
  echo "error: phase dir not found: phases/$PHASE" >&2
  echo "       run: ls phases/ to see valid phases" >&2
  exit 1
fi

if [[ -e "$LESSON_DIR" ]]; then
  echo "error: lesson already exists: phases/$PHASE/$LESSON" >&2
  exit 1
fi

if [[ ! "$LESSON" =~ ^[0-9]{2}-[a-z0-9-]+$ ]]; then
  echo "error: lesson slug must match NN-kebab-case (e.g. 03-tokenizers)" >&2
  exit 1
fi

mkdir -p "$LESSON_DIR/code" "$LESSON_DIR/notebook" "$LESSON_DIR/docs" "$LESSON_DIR/outputs"

PRETTY_TITLE="$TITLE"
if [[ -z "$PRETTY_TITLE" ]]; then
  PRETTY_TITLE="$(echo "${LESSON#[0-9][0-9]-}" | tr '-' ' ' | awk '{for (i=1; i<=NF; i++) $i=toupper(substr($i,1,1)) substr($i,2);}1')"
fi

PHASE_NUM="${PHASE%%-*}"
LESSON_NUM="${LESSON%%-*}"

cat >"$LESSON_DIR/docs/en.md" <<EOF
# $PRETTY_TITLE

> [شعار سطر واحد. الفكرة الأساسية التي تلتصق.]

**النوع:** بناء
** اللغات: ** بايثون
** المتطلبات الأساسية: ** [الدروس السابقة]
**الوقت:** ~75 دقيقة

## The Problem

[2-3 فقرات. ما الذي لا يستطيع المتعلم فعله بدون هذا؟ اجعلها ملموسة.]

## The Concept

[الحدس أولا. الرسوم البيانية والجداول والنماذج العقلية. لا يوجد رمز بعد.]

## Build It

### Step 1: [name]

[explanation]

\`\`\`python
# code here
\`\`\`

### Step 2: [name]

[explanation]

\`\`\`python
# code here
\`\`\`

## Use It

[كيف يحل الإطار الحقيقي نفس الشيء. قارن نسختك.]

## Ship It

[القطعة الأثرية القابلة لإعادة الاستخدام التي ينتجها هذا الدرس. حفظ في المخرجات/.]

## Exercises

1. [سهل - تعزيز المفهوم الأساسي]
2. [متوسط ​​- ينطبق على مشكلة مختلفة]
3. [صعب - يمتد أو يدمج مع الدروس السابقة]

## Key Terms

| مصطلح | ماذا يقول الناس | ماذا يعني في الواقع |
|------|----------------|----------------------|
|      |                |                      |

## Further Reading

- []() — []
EOF

cat >"$LESSON_DIR/code/main.py" <<'EOF'
التعريف الرئيسي ():
    رفع NotImplementedError("تنفيذ الدرس")


إذا كان __name__ == "__main__":
    main()
EOF

touch "$LESSON_DIR/notebook/.gitkeep"
touch "$LESSON_DIR/outputs/.gitkeep"

echo "المراحل التي تم إنشاؤها/$PHASE/$LESSON/"echo ""
echo "next:"
echo "  1. مراحل التحرير/$PHASE/$LESSON/docs/en.md"echo "  2. اكتب المراحل/$PHASE/$LESSON/code/main.py"echo "  3. أضف صف رابط تخفيض السعر إلى ROADMAP.md ضمن المرحلة $PHASE_NUM:"echo "     | $LESSON_NUM | [$PRETTY_TITLE](phases/$__TERM_0__/$__TERM_1__) | ✅ | ~75 دقيقة |"echo "  4. الالتزام الذري: git إضافة مراحل/$PHASE/$LESSON ROADMAP.md && git الالتزام \"feat(phase-$PHASE_NUM/$LESSON_NUM): $PRETTY_TITLE\""