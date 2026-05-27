#!/usr/bin/env node
/** * إنشاء سكريبت لموقع AI للهندسة من سكراتش. * يوزع README.md، ROADMAP.md، ومسرد/terms.md من جذر الريبو * ويقوم بإنشاء data.js مع جميع بيانات المرحلة/الدرس/المسرد. * * تشغيل: node site/build.js * يتم الاتصال بها تلقائيًا بواسطة GitHub الإجراءات في كل دفعة.
 */

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..');
const README_PATH = path.join(REPO_ROOT, 'README.md');
const ROADMAP_PATH = path.join(REPO_ROOT, 'ROADMAP.md');
const GLOSSARY_PATH = path.join(REPO_ROOT, 'glossary', 'terms.md');
const OUTPUT_PATH = path.join(__dirname, 'data.js');

const GITHUB_BASE = 'https://github.com/youssefupone-cpu/ai-engineering-from-scratch-ar/tree/main/';

// ─── تحليل ROADMAP.md لحالات الدرس ───────────────────────
function parseRoadmap(content) {
  const statuses = {}; // { "المرحلة 0": { حالة المرحلة، الدروس: { "بيئة التطوير": "مكتملة" } } }
  let currentPhase = null;
  let currentPhaseStatus = null;

  for (const line of content.split(/\r?\n/)) {
    // قم بمطابقة رؤوس المرحلة مثل: ## المرحلة 0: الإعداد والأدوات - ✅
    const phaseMatch = line.match(/^##\s+Phase\s+(\d+).*?—\s*(✅|🚧|⬚)/);
    if (phaseMatch) {
      const phaseId = parseInt(phaseMatch[1]);
      const statusEmoji = phaseMatch[2];
      currentPhaseStatus = statusEmoji === '✅' ? 'complete' : statusEmoji === '🚧' ? 'in-progress' : 'planned';
      currentPhase = `Phase ${phaseId}`;
      statuses[currentPhase] = { phaseStatus: currentPhaseStatus, lessons: {} };
      continue;
    }

    // قم بمطابقة صفوف الدرس مثل: | 01 | بيئة التطوير | ✅ |
    if (currentPhase) {
      const lessonMatch = line.match(/^\|\s*\d+\s*\|\s*(.+?)\s*\|\s*(✅|🚧|⬚)\s*\|/);
      if (lessonMatch) {
        const lessonName = lessonMatch[1].trim();
        const statusEmoji = lessonMatch[2];
        const status = statusEmoji === '✅' ? 'complete' : statusEmoji === '🚧' ? 'in-progress' : 'planned';
        statuses[currentPhase].lessons[lessonName] = status;
      }
    }
  }

  return statuses;
}

// ─── إعراب README.md للمراحل والدروس ───────────────────────
function parseReadme(content, roadmapStatuses) {
  const phases = [];

  // انقسم إلى كتل المرحلة
  // المرحلة 0 موجودة في كتلة <جدول>، والمراحل من 1 إلى 19 موجودة في كتل <تفاصيل>
  // سنقوم بتحليل سطرًا تلو الآخر لاستخراج رؤوس المراحل وجداول الدروس

  const lines = content.split(/\r?\n/);
  let currentPhase = null;
  let inLessonTable = false;
  let isCapstoneTable = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // مطابقة رأس المرحلة - تنسيقات متعددة مدعومة:
    // القديم: ### المرحلة 0: الإعداد والأدوات `12 lessons`
    // القديم: <summary><strong>المرحلة الأولى: أسس الرياضيات</strong> <code>22 درسًا</code>... <em>الوصف</em></summary>
    // جديد: ###![](https://img.shields.io/badge/Phase_0-Setup_&_Tooling-95A5A6?style=for-the-badge) `12 lessons`
    // جديد: <summary><b>🟣 المرحلة الأولى — أسس الرياضيات</b> <code>22 درسًا</code> <em>الوصف</em></summary>
    const phaseHeaderMatch =
      line.match(/###\s+Phase\s+(\d+):\s+(.+?)\s*`(\d+)\s+lessons?`/) ||
      line.match(/###\s+!\[\]\([^)]*?Phase[_\s]+(\d+)[-_]([^?)]+?)-[A-F0-9]{6}[^)]*\)\s*`(\d+)\s+lessons?`/i);
    const detailsHeaderMatch =
      line.match(/<summary><strong>Phase\s+(\d+):\s+(.+?)<\/strong>\s*<code>(\d+)\s+(?:lessons?|projects?)<\/code>.*?<em>(.*?)<\/em>/) ||
      line.match(/<summary>\s*<b>\s*(?:[^\w\s]+\s+)?Phase\s+(\d+)\s*[—\-:]\s*(.+?)<\/b>.*?<code>(\d+)\s+(?:lessons?|projects?)<\/code>.*?<em>(.*?)<\/em>/);

    if (phaseHeaderMatch) {
      const [, idStr, rawName] = phaseHeaderMatch;
      const id = parseInt(idStr);
      const name = rawName.replace(/_/g, ' ').trim();
      // ابحث عن الوصف في السطر التالي (اقتباس)
      let desc = '';
      for (let j = i + 1; j < Math.min(i + 5, lines.length); j++) {
        if (lines[j].startsWith('>')) {
          desc = lines[j].replace(/^>\s*/, '').trim();
          break;
        }
      }
      const roadmapKey = `Phase ${id}`;
      const phaseStatus = roadmapStatuses[roadmapKey]?.phaseStatus || 'planned';
      currentPhase = { id, name: name.trim(), status: phaseStatus, desc, lessons: [] };
      phases.push(currentPhase);
      inLessonTable = false;
      continue;
    }

    if (detailsHeaderMatch) {
      const [, idStr, name, , desc] = detailsHeaderMatch;
      const id = parseInt(idStr);
      const roadmapKey = `Phase ${id}`;
      const phaseStatus = roadmapStatuses[roadmapKey]?.phaseStatus || 'planned';
      currentPhase = { id, name: name.trim(), status: phaseStatus, desc: desc?.trim() || '', lessons: [] };
      phases.push(currentPhase);
      inLessonTable = false;
      continue;
    }

    // كشف بداية جدول الدرس
    if (currentPhase && line.match(/^\|\s*#\s*\|\s*Lesson/)) {
      inLessonTable = true;
      isCapstoneTable = false;
      continue;
    }

    // تخطي فاصل الجدول
    if (inLessonTable && line.match(/^\|[\s:|-]+\|$/)) {
      continue;
    }

    // تحليل صفوف الدرس
    if (inLessonTable && currentPhase && line.startsWith('|')) {
      // | 01 | [Dev Environment](phases/00-setup-and-tooling/01-dev-environment/) | بناء | بايثون، العقدة، Rust |
      // | 02 | شبكات متعددة الطبقات وتمرير أمامي | بناء | بايثون |
      const cols = line.split('|').map(c => c.trim()).filter(c => c.length > 0);
      if (cols.length >= 4) {
        const lessonCol = cols[1];
        const typeRaw = cols[2];
        const langRaw = cols[3];

        // قد يكون النوع عاديًا ("إنشاء") أو صورة درع:![Build](https://...))
        const typeBadgeMatch = typeRaw.match(/!\[([^\]]+)\]/);
        const type = typeBadgeMatch ? typeBadgeMatch[1] : typeRaw;

        // يمكن أن تكون اللغة الإنجليزية عادية ("Python، Rust") أو أعلام الرموز التعبيرية (🐍 🟦 🦀 🟣 ⚛️)
        const EMOJI_LANG = {
          '🐍': 'Python',
          '🟦': 'TypeScript',
          '🦀': 'Rust',
          '🟣': 'Julia',
          '⚛️': 'React',
          '⚛': 'React',
        };
        let lang = langRaw;
        if (/[\uD800-\uDBFF\u2600-\u27BF\u1F300-\u1FAFF]/.test(langRaw) || /[🐍🟦🦀🟣⚛]/u.test(langRaw)) {
          const tokens = Array.from(langRaw)
            .map(ch => EMOJI_LANG[ch])
            .filter(Boolean);
          if (tokens.length) lang = [...new Set(tokens)].join(', ');
          else if (langRaw.trim() === '—' || langRaw.trim() === '-') lang = '';
        }
        if (lang === '—' || lang === '-') lang = '';

        // تحقق مما إذا كان الدرس يحتوي على رابط (بمعنى أنه يحتوي على محتوى)
        const linkMatch = lessonCol.match(/\[(.+?)\]\((.+?)\)/);
        let lessonName, url;
        if (linkMatch) {
          lessonName = linkMatch[1];
          const relativePath = linkMatch[2];
          url = GITHUB_BASE + relativePath.replace(/^\//, '');
        } else {
          lessonName = lessonCol;
          url = null;
        }

        // الحصول على الحالة من خريطة الطريق
        const roadmapKey = `Phase ${currentPhase.id}`;
        const roadmapPhase = roadmapStatuses[roadmapKey];
        let status = 'planned';
        if (roadmapPhase) {
          // حاول العثور على الدرس المطابق عن طريق المطابقة الغامضة
          const lessonNameClean = lessonName.replace(/[-–—:]/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
          for (const [rName, rStatus] of Object.entries(roadmapPhase.lessons)) {
            const rNameClean = rName.replace(/[-–—:]/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
            if (rNameClean.includes(lessonNameClean) || lessonNameClean.includes(rNameClean) ||
                rNameClean.split(' ').slice(0, 3).join(' ') === lessonNameClean.split(' ').slice(0, 3).join(' ')) {
              status = rStatus;
              break;
            }
          }
        }

        // إذا كان يحتوي على رابط، فهو على الأقل مكتمل (تجاوز خريطة الطريق إذا لزم الأمر)
        if (url && status === 'planned') {
          status = 'complete';
        }

        // تستخدم جداول Capstone العمود الأوسط لرموز مرحلة المتطلبات الأساسية
        // (على سبيل المثال، "P11 P13 P14")، وليس تعداد البناء/التعلم. احتفظ بـ `type` على
        // إنشاء/تعلم المحور بحيث تظل محددات CSS (نوع البيانات = "إنشاء"/"تعلم")
        // صالح، وأصدر سلسلة المتطلبات المسبقة في حقل `combines` مخصص.
        const lessonEntry = {
          name: lessonName.trim(),
          status,
          type: isCapstoneTable ? 'Capstone' : type.trim(),
          lang: lang.trim() || '—',
          ...(isCapstoneTable && { combines: type.trim() }),
          ...(url && { url }),
        };
        currentPhase.lessons.push(lessonEntry);
      }
    }

    // نهاية الجدول
    if (inLessonTable && (line.match(/<\/td>/) || line.match(/<\/details>/) || (line.trim() === '' && i + 1 < lines.length && !lines[i + 1].startsWith('|')))) {
      inLessonTable = false;
    }

    // اكتشف أيضًا تنسيق الجدول النهائي (# | Project | Combines | Lang)
    if (currentPhase && line.match(/^\|\s*#\s*\|\s*Project/)) {
      inLessonTable = true;
      isCapstoneTable = true;
      continue;
    }
  }

  return phases;
}

// ─── استخرج ملخص الدرس + الكلمات الرئيسية من docs/en.md ───────────────
/** * قراءة لمرة واحدة لمستندات الدرس/en.md. * * العوائد: * الملخص - السطر الأول `> blockquote` (شعار الدرس ذو السطر الواحد). * الكلمات الرئيسية — جميع نصوص العناوين `### __TERM_0__` المرتبطة بـ ' · '. * H3 العناوين هي المفردات الأكثر كثافة في مستند الدرس * (على سبيل المثال، "منتج نقطي محدد الحجم · إخفاء سببي · KV ذاكرة تخزين مؤقت")، * لذا يقومون بتوسيع تغطية البحث دون تضخم data.js. * * كلا الحقلين عبارة عن سلاسل فارغة عندما يكون الملف غائباً أو لا يحتوي على أي شيء * المحتوى المطابق - متوقع للدروس المخططة بدون مستندات حتى الآن.
 */
function extractLessonMeta(relPath) {
  const docPath = path.join(REPO_ROOT, relPath, 'docs', 'en.md');
  const result = { summary: '', keywords: '' };
  try {
    const lines = fs.readFileSync(docPath, 'utf8').split(/\r?\n/);
    const h3s = [];
    for (const raw of lines) {
      const line = raw.trim();
      if (!result.summary && line.startsWith('> ') && line.length > 3) {
        const s = line.slice(2).trim();
        result.summary = s.length > 180 ? s.slice(0, 177) + '…' : s;
      }
      if (line.startsWith('### ')) {
        const heading = line.slice(4).trim();
        if (heading) h3s.push(heading);
      }
    }
    if (h3s.length) result.keywords = h3s.join(' · ');
  } catch (_) {
    // الملف غائب أو غير قابل للقراءة - متوقع للدروس المخططة.
  }
  return result;
}

// ─── تحليل المسرد/terms.md ─────────────────── ───────────────────
function parseGlossary(content) {
  const terms = [];
  let currentTerm = null;

  for (const line of content.split(/\r?\n/)) {
    // مطابقة رؤوس المصطلحات: ### الوكيل أو ### Adam (المُحسِّن)
    const termMatch = line.match(/^###\s+(.+)/);
    if (termMatch) {
      if (currentTerm && currentTerm.says && currentTerm.means) {
        terms.push(currentTerm);
      }
      currentTerm = { term: termMatch[1].trim(), says: '', means: '' };
      continue;
    }

    if (!currentTerm) continue;

    // طابق سطر "ما يقوله الناس".
    const saysMatch = line.match(/\*\*What people say:\*\*\s*"?(.+?)"?\s*$/);
    if (saysMatch) {
      currentTerm.says = saysMatch[1].replace(/^"/, '').replace(/"$/, '').trim();
      continue;
    }

    // طابق سطر "ما يعنيه هذا فعليًا".
    const meansMatch = line.match(/\*\*What it actually means:\*\*\s*(.+)/);
    if (meansMatch) {
      currentTerm.means = meansMatch[1].trim();
      continue;
    }
  }

  // دفع المصطلح الأخير
  if (currentTerm && currentTerm.says && currentTerm.means) {
    terms.push(currentTerm);
  }

  return terms;
}

// ─── اكتشف المخرجات / المصنوعات اليدوية (المهارات / المطالبات / الوكلاء) ──────────
function parseFrontmatter(text) {
  if (!text.startsWith('---')) return null;
  const end = text.indexOf('\n---', 4);
  if (end === -1) return null;
  const block = text.slice(4, end);
  const result = {};
  for (const raw of block.split(/\r?\n/)) {
    const line = raw.trimEnd();
    if (!line || line.startsWith('#') || !line.includes(':')) continue;
    const idx = line.indexOf(':');
    const key = line.slice(0, idx).trim();
    let value = line.slice(idx + 1).trim();
    if (value.startsWith('[') && value.endsWith(']')) {
      const inner = value.slice(1, -1).trim();
      result[key] = inner
        ? inner.split(',').map(s => s.trim().replace(/^['"]|['"]$/g, '')).filter(Boolean): []; } وإلا إذا ((value.startsWith('"') && value.endsWith('"')) || (القيمة. تبدأ مع ("'") && value.endsWith("'"))) { النتيجة[مفتاح] = value.slice(1, -1); } آخر { النتيجة [مفتاح] = القيمة؛ } } نتيجة الإرجاع؛
} وظيفة اكتشاف التحف () { القطع الأثرية الثابتة = []؛ constphaseDir = path.join(REPO_ROOT, 'phases'); إذا قام (!fs.existsSync(phasesDir)) بإرجاع القطع الأثرية؛ const VALID_TYPES = ['skill', 'prompt', 'agent']; لـ (constphaseDirName of fs.readdirSync(phasesDir).sort()) { constphaseMatch =phaseDirName.match(/^([0-9]{2})-([a-z0-9-]+)$/); إذا استمر (!phaseMatch)؛ constphaseId = parseInt(phaseMatch[1], 10); constphaseDir = path.join(phasesDir,phaseDirName); لـ (const LessonDirName of fs.readdirSync(phaseDir).sort()) { const LessonMatch = LessonDirName.match(/^([0-9]{2})-([a-z0-9-]+)$/); إذا استمر (!lessonMatch)؛ const LessonId = parseInt(lessonMatch[1], 10); const LessonRel = `phases/${phaseDirName}/${lessonDirName}`; constputsDir = path.join(phaseDir, LessonDirName, 'outputs'); إذا (fs.existsSync(outputsDir)) { لـ (ملف const الخاص بـ fs.readdirSync(outputsDir).sort()) { إذا استمر (! file.endsWith('.md')))؛ const الجذعية = file.replace(/\.md$/, ''); نوع ثابت = VALID_TYPES.find(t => الجذعية.startsWith(`${t}-`)); إذا (!اكتب) تابع؛ دع التعريف = {}; حاول { meta = parseFrontmatter(fs.readFileSync(path.join(outputsDir, file), 'utf8')) || {}; } أمسك (_) {} التحف.push({ نوع: نوع، الاسم: (meta.name || الجذعية).trim()، الوصف: (meta.description || '').trim(), العلامات: Array.isArray(meta.tags)؟ العلامات الوصفية: []، المرحلة: معرف المرحلة، الدرس: معرف الدرس، مسار الدرس: الدرس، الملف: `${lessonRel}/outputs/${file}`، }); } } const MissionPath = path.join(phaseDir, LessonDirName, 'mission.md'); إذا (fs.existsSync(missionPath)) { دع الخط الأول = ''; حاول { firstLine = fs.readFileSync(missionPath, 'utf8').split(/\r?\n/)[0].replace(/^#\s+/, '').trim(); } أمسك (_) {} التحف.push({ النوع: "مهمة"، الاسم: الخط الأول || `مهمة ${lessonDirName}`، الوصف: ''، العلامات: []، المرحلة: معرف المرحلة، الدرس: معرف الدرس، مسار الدرس: الدرس، الملف: `${lessonRel}/mission.md`، }); } } } عودة القطع الأثرية.
} // ─── البناء الرئيسي ───────────────────────── ─────────────────────────
بناء الوظيفة () { console.log('📖 قراءة الملفات المصدر...'); const readme = fs.readFileSync(README_PATH, 'utf8'); خريطة طريق const = fs.readFileSync(ROADMAP_PATH, 'utf8'); مسرد const = fs.readFileSync(GLOSSARY_PATH, 'utf8'); console.log('🔍 تحليل ROADMAP.md...'); const roadmapStatuses = parseRoadmap(roadmap); console.log('🔍 تحليل README.md...'); constphase = parseReadme(readme, roadmapStatuses); console.log('🔍 إعراب المسرد/terms.md...'); constlossaryTerms = parseGlossary(glossary); console.log('🔍 اكتشاف المخرجات + مهام المرحلة 14...'); قطع أثرية ثابتة = DiscoverArtifacts(); console.log('📚 استخراج ملخصات الدروس + الكلمات الرئيسية من docs/en.md...'); دع تلخيصها = 0، مع الكلمات الرئيسية = 0؛ لـ (المرحلة الثابتة من المراحل) { for (درس ثابت من المرحلة. الدروس) { إذا (الدرس.url) { const relPath = Lesson.url.replace(GITHUB_BASE, '').replace(/\/+$/, '');const meta = extractLessonMeta(relPath); إذا (meta.summary) { Lesson.summary = meta.summary؛ تلخيص++; } إذا (meta.keywords) { Lesson.keywords = meta.keywords؛ withKeywords++; } } } } // احصائيات دع مجموع الدروس = 0؛ دع CompleteLessons = 0؛ مراحل.forEach(ع => { TotalLessons += p.lessons.length; CompleteLessons += p.lessons.filter(l => l.status === 'Complete').length; }); console.log(`\n📊 الإحصائيات:`); console.log(`المراحل: ${phases.length}`); console.log(` الدروس: ${totalLessons}`); console.log(` مكتمل: ${CompleteLessons}`); console.log(` الملخصات: ${summarized}, الكلمات الرئيسية: ${withKeywords}`); console.log(` مصطلحات المسرد: ${glossaryTerms.length}`); console.log(` القطع الأثرية: ${artifacts.length}`); // إنشاء data.js إخراج const = `// تم إنشاؤه تلقائيًا بواسطة build.js - لا تقم بالتحرير يدويًا.
// آخر بناء: ${new Date().toISOString()} const PHASES = ${JSON.stringify(phases, null, 2)}; const GLOSSARY = ${JSON.stringify(glossaryTerms, null, 2)}; const ARTIFACTS = ${JSON.stringify(artifacts, null, 2)};
`; fs.writeFileSync(OUTPUT_PATH, الإخراج, 'utf8'); console.log(`\n✅ تم إنشاؤه ${OUTPUT_PATH}`);
} يبني()؛"