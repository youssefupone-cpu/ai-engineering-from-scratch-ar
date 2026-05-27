/** * تعقب التقدم المحلي فقط. * * يخزن كل شيء في المتصفح الخاص بالمستخدم (localStorage). لا يوجد شبكة، * لا يوجد حساب ولا خادم. البيانات لا تترك الجهاز أبدًا. * * المخطط (تم إصداره حتى نتمكن من الترحيل لاحقًا دون استخدام الأسلحة النووية): * * aifs:التقدم:v1 = { * الدروس: { * "<مسار الدرس>": { * الإجابات: { "<qid>": { منتقى: رقم، صحيح: منطقي، t: رقم } }، * اكتمل في: رقم | لاغية, * تمت الزيارة في: الرقم * } * }، * تم التحديث في: الرقم * } * * "<lesson-path>" يتطابق مع المسار المستخدم في Lesson.html?path=... وin * عناوين URL الخاصة بـ data.js (على سبيل المثال، "phases/00-setup-and-tooling/01-dev-environment"). * * "<qid>" هو "<stage>-q<index>" على سبيل المثال. "pre-q0"، لمطابقة عارض الاختبار.
 */
(function () {
  var STORAGE_KEY = 'aifs:progress:v1';
  var listeners = [];

  function emptyState() {
    return { lessons: {}, updatedAt: 0 };
  }

  function read() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return emptyState();
      var parsed = JSON.parse(raw);
      if (!parsed || typeof parsed !== 'object' || !parsed.lessons) return emptyState();
      return parsed;
    } catch (e) {
      return emptyState();
    }
  }

  function write(state) {
    state.updatedAt = Date.now();
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (e) {
      // الحصة أو التخزين المعطل؛ تفشل بصمت
    }
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  }

  function ensureLesson(state, path) {
    if (!state.lessons[path]) {
      state.lessons[path] = { answers: {}, completedAt: null, visitedAt: 0 };
    }
    return state.lessons[path];
  }

  function recordVisit(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.visitedAt = Date.now();
    write(state);
  }

  function recordAnswer(path, qid, picked, correct) {
    if (!path || !qid) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    lesson.answers[qid] = { picked: picked, correct: !!correct, t: Date.now() };
    write(state);
  }

  function markLessonComplete(path) {
    if (!path) return;
    var state = read();
    var lesson = ensureLesson(state, path);
    if (!lesson.completedAt) {
      lesson.completedAt = Date.now();
      write(state);
    }
  }

  function unmarkLessonComplete(path) {
    if (!path) return;
    var state = read();
    if (state.lessons[path] && state.lessons[path].completedAt) {
      state.lessons[path].completedAt = null;
      write(state);
    }
  }

  function getLessonProgress(path) {
    if (!path) return null;
    var state = read();
    return state.lessons[path] || { answers: {}, completedAt: null, visitedAt: 0 };
  }

  function isLessonComplete(path) {
    var lp = getLessonProgress(path);
    return !!(lp && lp.completedAt);
  }

  /** * في ضوء قائمة عناوين URL الخاصة بالدروس (عناوين URL الكاملة GitHub من data.js)، قم بإحصاء كيفية القيام بذلك * أكمل العديد من المستخدم. قم بالمطابقة حسب المسار "المراحل/.../..." اللاحق.
   */
  function countCompletedFromUrls(urls) {
    var state = read();
    var n = 0;
    for (var i = 0; i < urls.length; i++) {
      var path = extractPath(urls[i]);
      if (path && state.lessons[path] && state.lessons[path].completedAt) n++;
    }
    return n;
  }

  function extractPath(url) {
    if (!url) return '';
    var m = String(url).match(/(phases\/[^/]+\/[^/]+)\/?/);
    return m ? m[1] : '';
  }

  function totalCompleted() {
    var state = read();
    var n = 0;
    for (var k in state.lessons) {
      if (state.lessons[k].completedAt) n++;
    }
    return n;
  }

  function reset() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](emptyState()); } catch (_) {}
    }
  }

  function onChange(fn) {
    if (typeof fn === 'function') listeners.push(fn);
  }

  // المزامنة بين علامات التبويب: إذا قام المستخدم بمسح أو تحديث التقدم في علامة تبويب أخرى،
  // قم بتحديث المستمعين هنا أيضًا.
  window.addEventListener('storage', function (e) {
    if (e.key !== STORAGE_KEY) return;
    var state = read();
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](state); } catch (_) {}
    }
  });

  window.AIFSProgress = {
    recordVisit: recordVisit,
    recordAnswer: recordAnswer,
    markLessonComplete: markLessonComplete,
    unmarkLessonComplete: unmarkLessonComplete,
    getLessonProgress: getLessonProgress,
    isLessonComplete: isLessonComplete,
    countCompletedFromUrls: countCompletedFromUrls,
    extractPath: extractPath,
    totalCompleted: totalCompleted,
    reset: reset,
    onChange: onChange,
  };
})();
