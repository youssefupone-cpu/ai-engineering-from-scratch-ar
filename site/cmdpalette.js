/** * لوحة الأوامر - بحث عام يتم تشغيله بواسطة Cmd/Ctrl+K أو زر البحث. * * يبحث في عناوين الدروس، والملخصات، وأسماء المراحل، واللغات، والأنواع، و * مصطلحات المسرد من جانب العميل بالكامل من البيانات التي تم تحميلها بالفعل في data.js. * لا توجد طلبات الشبكة. لا تبعيات خارجية. * * API (مرفق بـ window.CmdPalette): * CmdPalette.open() — افتح اللوحة * CmdPalette.Close() — أغلق اللوحة * * أزرار التشغيل: أي عنصر له خاصية [data-cmd-palette].
 */
(function () {
  'استخدام صارم';

  // ── الثوابت ────────────────────────── ──────────────────────────
  var PALETTE_ID  = 'cmdPalette';
  var MAX_RESULTS = 12;
  var BODY_ATTR   = 'data-palette-open';

  // ── حالة الوحدة النمطية ──────────────────────── ─────────────────────────
  var _index      = null;   // مجموعة مسطحة من العناصر القابلة للبحث
  var _activeIdx  = -1;
  var _isOpen     = false;
  var _prevFocus  = null;

  // ── فهرس البحث ──────────────────────── ─────────────────────────
  /** * أنشئ فهرس البحث المسطح مرة واحدة من window.PHASES وwindow.GLOSSARY. * Idempotent: تُرجع الاستدعاءات اللاحقة المصفوفة المخزنة مؤقتًا.
   */
  function buildIndex() {
    if (_index !== null) return _index;
    _index = [];

    if (typeof PHASES !== 'undefined' && Array.isArray(PHASES)) {
      for (var i = 0; i < PHASES.length; i++) {
        var phase = PHASES[i];
        for (var j = 0; j < phase.lessons.length; j++) {
          var lesson = phase.lessons[j];

          // قم باستخراج المراحل/…/… المسار المستخدم للدرس.html?path=
          var lessonPath = '';
          if (lesson.url) {
            var m = lesson.url.match(/(phases\/[^/?#]+\/[^/?#]+)/);
            if (m) lessonPath = m[1];
          }

          _index.push({
            kind:       'lesson',
            id:         'l:' + i + ':' + j,
            phaseId:    phase.id,
            phaseName:  phase.name,
            name:       lesson.name     || '',
            summary:    lesson.summary  || '',
            keywords:   lesson.keywords || '',
            type:       lesson.type     || '',
            lang:       lesson.lang     || '',
            status:     lesson.status   || '',
            lessonPath: lessonPath,
            url:        lesson.url      || '',
          });
        }
      }
    }

    if (typeof GLOSSARY !== 'undefined' && Array.isArray(GLOSSARY)) {
      for (var k = 0; k < GLOSSARY.length; k++) {
        var g = GLOSSARY[k];
        _index.push({
          kind:    'glossary',
          id:      'g:' + k,
          name:    g.term  || '',
          summary: g.means || '',
          says:    g.says  || '',
        });
      }
    }

    if (typeof ARTIFACTS !== 'undefined' && Array.isArray(ARTIFACTS)) {
      for (var a = 0; a < ARTIFACTS.length; a++) {
        var art = ARTIFACTS[a];
        _index.push({
          kind:       'artifact',
          id:         'a:' + a,
          artKind:    art.kind || 'artifact',
          name:       art.name || '',
          summary:    art.description || '',
          keywords:   Array.isArray(art.tags) ? art.tags.join(' ') : '',
          phaseId:    art.phase,
          lesson:     art.lesson,
          lessonPath: art.lessonPath || '',
          file:       art.file || '',
        });
      }
    }

    return _index;
  }

  // ── التهديف ─────────────────────────── ───────────────────────────
  function scoreItem(item, q) {
    // q مكتوب بالفعل بأحرف صغيرة + تم قطعه بواسطة المتصل
    var name     = item.name.toLowerCase();
    var summary  = (item.summary  || '').toLowerCase();
    var keywords = (item.keywords || '').toLowerCase();
    var phase    = (item.phaseName || '').toLowerCase();
    var lang     = (item.lang  || '').toLowerCase();
    var type     = (item.type  || '').toLowerCase();
    var says     = (item.says  || '').toLowerCase();

    var s = 0;

    // المطابقة التامة للاسم الكامل — الأولوية القصوى
    if (name === q) return 200;

    // تطابقات السلسلة الفرعية في الاسم (الإشارة الأكثر أهمية)
    if (name.startsWith(q))          s += 100;
    else if (name.indexOf(q) !== -1) s +=  70;

    // استعلام متعدد الكلمات: يجب أن تظهر كل كلمة في مكان ما في الاسم
    var words = q.split(/\s+/).filter(Boolean);
    if (words.length > 1) {
      var allInName = words.every(function (w) { return name.indexOf(w) !== -1; });
      if (allInName) {
        s += (s === 0 ? 65 : 20);
      } else {
        // الأضعف: كل كلمة منتشرة عبر الاسم + الملخص + الكلمات الرئيسية + المرحلة
        var blob = name + ' ' + summary + ' ' + keywords + ' ' + phase;
        var allInBlob = words.every(function (w) { return blob.indexOf(w) !== -1; });
        if (allInBlob) s += 15;
      }
    }

    // الحقول الداعمة - مرتبة حسب الصلة المتوقعة
    if (summary.indexOf(q)  !== -1) s += 25;
    if (keywords.indexOf(q) !== -1) s += 22; // H3 العناوين: مفردات كثيفة
    if (says.indexOf(q)     !== -1) s += 22; // معجم "ما يقوله الناس"
    if (phase.indexOf(q)    !== -1) s += 18;
    if (lang.indexOf(q)     !== -1) s += 14;
    if (type.indexOf(q)     !== -1) s += 10;

    // احتياطي كلمة واحدة: تطابق بادئة حد الكلمة على الرموز المميزة للاسم
    if (s === 0 && words.length === 1) {
      var nameParts = name.split(/[\s\-–—:,]+/).filter(Boolean);
      for (var i = 0; i < nameParts.length; i++) {
        if (nameParts[i].startsWith(q)) { s += 30; break; }
      }
      // الملاذ الأخير: كلمة واحدة في أي مكان في الكلمات الرئيسية أو الملخص
      if (s === 0 && keywords.indexOf(q) !== -1) s += 18;
      if (s === 0 && summary.indexOf(q)  !== -1) s += 12;
    }

    return s;
  }

  function search(query) {
    var q = query.trim().toLowerCase();
    if (!q) return [];

    var items   = buildIndex();
    var results = [];

    for (var i = 0; i < items.length; i++) {
      var s = scoreItem(items[i], q);
      if (s > 0) results.push({ item: items[i], s: s });
    }

    results.sort(function (a, b) { return b.s - a.s; });
    return results.slice(0, MAX_RESULTS).map(function (r) { return r.item; });
  }

  // ── المرافق ────────────────────────── ──────────────────────────
  function escHtml(str) {
    var d = document.createElement('div');
    d.textContent = (str == null) ? '' : String(str);
    return d.innerHTML;
  }

  /** * قم بتمييز الظهور الأول لـ `query` (أو أول كلمة مطابقة له) * داخل `text`. تُرجع سلسلة HTML آمنة مع علامة <mark> حول المطابقة.
   */
  function highlight(text, query) {
    if (!text) return '';
    if (!query) return escHtml(text);

    var lower = text.toLowerCase();
    var q     = query.trim().toLowerCase();
    var idx   = lower.indexOf(q);
    var matchLen = q.length;

    if (idx === -1) {
      // جرب كل كلمة على حدة
      var words = q.split(/\s+/).filter(Boolean);
      for (var i = 0; i < words.length; i++) {
        idx = lower.indexOf(words[i]);
        if (idx !== -1) { matchLen = words[i].length; break; }
      }
    }

    if (idx === -1) return escHtml(text);

    return (
      escHtml(text.slice(0, idx)) +
      '<mark>' + escHtml(text.slice(idx, idx + matchLen)) + '</mark>' +
      escHtml(text.slice(idx + matchLen))
    );
  }

  function truncate(str, max) {
    if (!str || str.length <= max) return str || '';
    var cut = str.slice(0, max).replace(/\s+\S*$/, '');
    return (cut.length > max * 0.6 ? cut : str.slice(0, max)) + '…';
  }

  // ── لوحة DOM (تم إنشاؤها بتكاسل عند الفتح الأول) ─────────────────────
  function createPaletteDOM() {
    if (document.getElementById(PALETTE_ID)) return;

    // كشف النظام الأساسي لتلميح اختصار التذييل
    var isMac = /Mac|iPhone|iPod|iPad/.test(
      (navigator.userAgentData && navigator.userAgentData.platform) ||
      navigator.platform || ''
    );
    var shortcutLabel = isMac ? '⌘K' : 'Ctrl+K';

    var el = document.createElement('div');
    el.id = PALETTE_ID;
    el.setAttribute('role', 'dialog');
    el.setAttribute('aria-modal', 'true');
    el.setAttribute('aria-label', 'بحث في الدروس والمسرد');

    el.innerHTML =
      '<div class="cp-backdrop" id="cpBackdrop"></div>' +
      '<div class="cp-panel">' +
        '<div class="cp-search-row">' +
          '<svg class="cp-search-icon" width="16" height="16" viewBox="0 0 24 24"' +
          'ملء = "لا شيء" الحد = "اللون الحالي" عرض الحد = "2.5"' +
          'السكتة الدماغية-linecap = "جولة" السكتة الدماغية - Linejoin = "جولة" aria-hidden = "صحيح">' +
            '<circle cx="11" cy="11" r="8"/>' +
            '<line x1="21" y1="21" x2="16.65" y2="16.65"/>' +
          '</svg>' +
          '<input class="cp-input" id="cpInput" type="search"' +
          'placeholder="البحث في الدروس والمسرد..."' +
          'الإكمال التلقائي = "إيقاف" التصحيح التلقائي = "إيقاف"' +
          'تكبير تلقائي = "إيقاف" التدقيق الإملائي = "خطأ"' +
          'aria-label = "بحث" aria-autocomplete = "قائمة"' +
          ' aria-controls="cpResults">' +
          '<kbd class="cp-kbd-esc" id="cpKbdEsc">Esc</kbd>' +
        '</div>' +
        '<ul class="cp-results" id="cpResults"' +
        'role="listbox" aria-label="نتائج البحث"></ul>' +
        '<div class="cp-footer">' +
          '<span class="cp-footer-group">' +
            '<kbd>↑</kbd><kbd>↓</kbd>' +
            '<span class="cp-footer-label">navigate</span>' +
          '</span>' +
          '<span class="cp-footer-group">' +
            '<kbd>↵</kbd>' +
            '<span class="cp-footer-label">open</span>' +
          '</span>' +
          '<span class="cp-footer-group">' +
            '<kbd>Esc</kbd>' +
            '<span class="cp-footer-label">close</span>' +
          '</span>' +
          '<span class="cp-footer-shortcut">' + shortcutLabel + '</span>' +
        '</div>' +
      '</div>';

    document.body.appendChild(el);

    // ربط التفاعلات الداخلية
    document.getElementById('cpBackdrop').addEventListener('click', close);
    document.getElementById('cpKbdEsc').addEventListener('click', close);

    var inp = document.getElementById('cpInput');
    inp.addEventListener('input', _onInput);
    inp.addEventListener('keydown', _onKeyDown);
  }

  function _palEl()   { return document.getElementById(PALETTE_ID); }
  function _inputEl() { return document.getElementById('cpInput'); }
  function _listEl()  { return document.getElementById('cpResults'); }

  // ── فتح / إغلاق ──────────────────────── ─────────────────────────
  function open() {
    if (_isOpen) {
      // مفتوح بالفعل — make تأكد من تركيز الإدخال
      var inp = _inputEl();
      if (inp) inp.focus();
      return;
    }

    _prevFocus = document.activeElement || null;
    _isOpen    = true;
    _activeIdx = -1;

    createPaletteDOM();
    document.body.setAttribute(BODY_ATTR, '');

    // تأخير إطارين: الإطار الأول يؤدي إلى الانتقال، والثاني يضمن التركيز
    requestAnimationFrame(function () {
      var pal = _palEl();
      if (pal) pal.classList.add('cp-open');

      requestAnimationFrame(function () {
        var inp = _inputEl();
        if (inp) {
          inp.focus();
          var q = inp.value.trim();
          renderResults(q ? search(q) : []);
        }
      });
    });
  }

  function close() {
    if (!_isOpen) return;
    _isOpen    = false;
    _activeIdx = -1;

    var pal = _palEl();
    if (pal) pal.classList.remove('cp-open');
    document.body.removeAttribute(BODY_ATTR);

    // إعادة التركيز إلى المكان الذي كان فيه المستخدم من قبل
    try {
      if (_prevFocus && typeof _prevFocus.focus === 'function') {
        _prevFocus.focus();
      }
    } catch (_) { /* ربما تمت إزالة العنصر من DOM */ }
    _prevFocus = null;
  }

  // ── عرض النتائج ─────────────────────── ────────────────────────
  function renderResults(results) {
    var list = _listEl();
    if (!list) return;

    var query = (_inputEl() ? _inputEl().value : '').trim();

    if (!query) {
      list.innerHTML =
        '<li class="cp-empty" role="option" aria-disabled="true">' +
        'اكتب للبحث في 435 درسًا و489 مخرجًا ومصطلحات المصطلحات' +
        '</li>';
      _activeIdx = -1;
      return;
    }

    if (results.length === 0) {
      list.innerHTML =
        '<li class="cp-empty" role="option" aria-disabled="true">' +
        'لا توجد نتائج لـ <em>' + escHtml(query) + '</em>' +
        '</li>';
      _activeIdx = -1;
      return;
    }

    var html = '';
    for (var i = 0; i < results.length; i++) {
      var r    = results[i];
      var dest = '';
      var chip = '';
      var chipClass = 'cp-item-chip';

      if (r.kind === 'lesson') {
        // تفضيل القارئ الموجود في الموقع؛ الرجوع إلى GitHub URL
        dest = r.lessonPath
          ? 'lesson.html?path=' + encodeURIComponent(r.lessonPath)
          : r.url;
        chip = 'المرحلة ' + String(r.phaseId).padStart(2, '0');
      } else if (r.kind === 'artifact') {
        // انتقل إلى الدرس الذي أنتج هذه القطعة الأثرية
        dest = r.lessonPath
          ? 'lesson.html?path=' + encodeURIComponent(r.lessonPath)
          : ('https://github.com/rohitg00/ai-engineering-from-scratch/tree/main/' + r.file);
        var ak = (r.artKind || 'artifact');
        chip = ak.charAt(0).toUpperCase() + ak.slice(1);
        chipClass += ' cp-item-chip--alt';
      } else {
        // الارتباط العميق: قم بتعبئة بحث المسرد مسبقًا باسم المصطلح الدقيق
        // لذلك ينتقل المستخدم مباشرة إلى التعريف، وليس القائمة الكاملة.
        dest      = 'glossary.html?q=' + encodeURIComponent(r.name);
        chip      = 'المعجم';
        chipClass += ' cp-item-chip--alt';
      }

      var snippet = r.summary ? truncate(r.summary, 110) : '';
      var metaParts = [];
      if (r.kind === 'lesson') {
        if (r.type && r.type !== '—') metaParts.push(r.type);
        if (r.lang && r.lang !== '—') metaParts.push(r.lang);
      } else if (r.kind === 'artifact') {
        if (r.phaseId !== undefined && r.phaseId !== null) {
          metaParts.push('المرحلة ' + String(r.phaseId).padStart(2, '0'));
        }
      }
      var meta = metaParts.join(' · '); // ·

      html +=
        '<li class="cp-item" role="option" aria-selected="false"' +
        ' data-idx="' + i + '"' +
        ' data-href="' + escHtml(dest) + '">' +
          '<div class="cp-item-body">' +
            '<span class="' + chipClass + '">' + escHtml(chip) + '</span>' +
            '<span class="cp-item-name">'    + highlight(r.name,    query) + '</span>' +
            (snippet ? '<span class="cp-item-summary">' + highlight(snippet, query) + '</span>' : '') +
            (meta    ? '<span class="cp-item-meta">'    + escHtml(meta)             + '</span>' : '') +
          '</div>' +
          '<svg class="cp-item-arrow" width="12" height="12" viewBox="0 0 24 24"' +
          'ملء = "لا شيء" الحد = "اللون الحالي" عرض الحد = "2"' +
          'السكتة الدماغية-linecap = "جولة" السكتة الدماغية - Linejoin = "جولة" aria-hidden = "صحيح">' +
            '<polyline points="9 18 15 12 9 6"/>' +
          '</svg>' +
        '</li>';
    }

    list.innerHTML = html;
    _activeIdx = -1;

    // إرفاق معالجات التفاعل
    var items = list.querySelectorAll('.cp-item');
    for (var j = 0; j < items.length; j++) {
      items[j].addEventListener('click',     _onItemClick);
      items[j].addEventListener('mousemove', _onItemMouseMove);
    }
  }

  // ── معالجات الأحداث ─────────────────────── ────────────────────────
  function _onInput(e) {
    var query = e.target.value;
    renderResults(search(query));
    _activeIdx = -1;
  }

  function _onKeyDown(e) {
    var list  = _listEl();
    var items = list ? list.querySelectorAll('.cp-item') : [];
    var count = items.length;

    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!count) return;
        _activeIdx = (_activeIdx + 1) % count;
        _updateActive(items);
        break;

      case 'ArrowUp':
        e.preventDefault();
        if (!count) return;
        _activeIdx = (_activeIdx - 1 + count) % count;
        _updateActive(items);
        break;

      case 'Enter': {
        e.preventDefault();
        const target = (_activeIdx >= 0 && items[_activeIdx])
          ? items[_activeIdx]
          : (count === 1 ? items[0] : null);
        if (target) _navigate(target);
        break;
      }

      case 'Tab':
        // ملائمة التركيز داخل اللوحة (العنصر التفاعلي فقط هو الإدخال)
        e.preventDefault();
        break;

      case 'Escape':
        e.preventDefault();
        close();
        break;
    }
  }

  function _updateActive(items) {
    for (var i = 0; i < items.length; i++) {
      var active = (i === _activeIdx);
      items[i].classList.toggle('cp-item--active', active);
      items[i].setAttribute('aria-selected', active ? 'true' : 'false');
      if (active) items[i].scrollIntoView({ block: 'nearest' });
    }
  }

  function _onItemClick(e) {
    _navigate(e.currentTarget);
  }

  function _onItemMouseMove(e) {
    var list = _listEl();
    if (!list) return;
    var idx = parseInt(e.currentTarget.getAttribute('data-idx'), 10);
    if (idx !== _activeIdx) {
      _activeIdx = idx;
      _updateActive(list.querySelectorAll('.cp-item'));
    }
  }

  function _navigate(item) {
    var href = item.getAttribute('data-href');
    if (!href) return;
    close();
    window.location.href = href;
  }

  // ── اختصار لوحة المفاتيح العامة (Cmd/Ctrl+K) ───────────────────────
  document.addEventListener('keydown', function (e) {
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      if (_isOpen) {
        // لوحة الألوان مفتوحة بالفعل — ما عليك سوى إعادة تركيز الإدخال
        var inp = _inputEl();
        if (inp) inp.focus();
      } else {
        open();
      }
    }
  });

  // ── Init: أزرار تشغيل الأسلاك + إنشاء الفهرس بفارغ الصبر ─────────────────
  function _init() {
    // أي عنصر به [data-cmd-palette] يفتح اللوحة عند النقر عليه
    var triggers = document.querySelectorAll('[data-cmd-palette]');
    for (var i = 0; i < triggers.length; i++) {
      triggers[i].addEventListener('click', function (e) {
        e.preventDefault();
        open();
      });
    }

    // أنشئ فهرس البحث الآن بحيث تكون أول ضغطة على المفتاح فورية
    buildIndex();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _init);
  } else {
    _init();
  }

  // ── عام API ────────────────────────── ──────────────────────────
  window.CmdPalette = { open: open, close: close };

}());
