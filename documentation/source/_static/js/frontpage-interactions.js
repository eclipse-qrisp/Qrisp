/* =============================================================================
   Frontpage interactions — scroll reveal, card tilt, code typing
   ============================================================================= */
(function () {
  'use strict';

  /* ==========================================================
     1.  SCROLL-TRIGGERED FADE / SLIDE-IN
     ========================================================== */
  function initScrollReveal() {
    var targets = [];

    function addSectionHeader(sectionId) {
      var section = document.getElementById(sectionId);
      if (!section) return;
      var heading = section.querySelector('h1, h2');
      var divider = section.querySelector('.section-divider');
      var subtitle = section.querySelector('.section-subtitle');
      [heading, divider, subtitle].forEach(function (el) {
        if (el) {
          el.classList.add('scroll-reveal');
          targets.push(el);
        }
      });
    }

    function addStaggeredChildren(sectionId, childSelector, staggerMs) {
      var section = document.getElementById(sectionId);
      if (!section) return;
      var children = section.querySelectorAll(childSelector);
      children.forEach(function (el, i) {
        el.classList.add('scroll-reveal');
        el.style.transitionDelay = (i * (staggerMs || 80)) + 'ms';
        targets.push(el);
      });
    }

    function addContentBlock(sectionId, selector) {
      var section = document.getElementById(sectionId);
      if (!section) return;
      var els = selector ? section.querySelectorAll(selector) : [section];
      els.forEach(function (el) {
        el.classList.add('scroll-reveal');
        targets.push(el);
      });
    }

    /* Key Features */
    addSectionHeader('key-features');
    addStaggeredChildren('key-features', '.sd-col', 80);

    /* Code Comparison */
    addSectionHeader('dive-into-qrisp-code');
    addContentBlock('dive-into-qrisp-code', '.code-example-text');

    /* Partners */
    addSectionHeader('who-is-behind-qrisp');
    addContentBlock('who-is-behind-qrisp', '.code-example-text');
    addStaggeredChildren('who-is-behind-qrisp', '.sd-col', 60);

    if (!targets.length) return;

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('revealed');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

    targets.forEach(function (el) { observer.observe(el); });
  }

  /* ==========================================================
     2.  KEY-FEATURE CARD TILT + GLOW
     ========================================================== */
  function initCardTilt() {
    var section = document.getElementById('key-features');
    if (!section) return;

    var cards = section.querySelectorAll('.sd-card');
    cards.forEach(function (card) {
      /* Ensure positioning context for the glow overlay */
      card.style.position = 'relative';
      card.style.overflow = 'hidden';

      var glow = document.createElement('div');
      glow.className = 'tilt-glow';
      card.appendChild(glow);

      card.addEventListener('mousemove', function (e) {
        var rect = card.getBoundingClientRect();
        var x = e.clientX - rect.left;
        var y = e.clientY - rect.top;
        var cx = rect.width / 2;
        var cy = rect.height / 2;

        var rotX = ((y - cy) / cy) * -4;
        var rotY = ((x - cx) / cx) * 4;

        card.style.transform =
          'perspective(800px) rotateX(' + rotX + 'deg) rotateY(' + rotY + 'deg) translateY(-5px)';
        glow.style.background =
          'radial-gradient(circle at ' + x + 'px ' + y + 'px, rgba(0,140,255,0.07) 0%, transparent 55%)';
      });

      card.addEventListener('mouseleave', function () {
        card.style.transform = '';
        glow.style.background = '';
      });
    });
  }

  /* ==========================================================
     3.  CODE TYPING ANIMATION  (Qrisp column only)
     Uses an overlay mask so the original syntax-highlighted
     HTML is never modified — Pygments highlighting is preserved.
     ========================================================== */
  function initTypingAnimation() {
    var section = document.getElementById('dive-into-qrisp-code');
    if (!section) return;

    /* Locate the Qrisp <pre> — second <td> in the code comparison table */
    var qrispPre = null;
    var tables = section.querySelectorAll('table');
    for (var t = 0; t < tables.length; t++) {
      var tds = tables[t].querySelectorAll('td');
      if (tds.length >= 2) {
        var pre = tds[1].querySelector('pre');
        if (pre) { qrispPre = pre; break; }
      }
    }
    if (!qrispPre) return;

    /* Compute line metrics */
    var style = window.getComputedStyle(qrispPre);
    var lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.4;
    var preHeight = qrispPre.scrollHeight;
    var totalLines = Math.round(preHeight / lineHeight);
    if (totalLines < 2) return;

    /* Ensure the pre has positioning context */
    var pos = style.position;
    if (pos === 'static' || pos === '') qrispPre.style.position = 'relative';
    qrispPre.style.overflow = 'hidden';

    /* Get the background colour so the overlay blends seamlessly */
    var bg = style.backgroundColor;
    if (!bg || bg === 'transparent' || bg === 'rgba(0, 0, 0, 0)') bg = '#f8f8f8';

    /* Create the overlay mask — sits atop the code and shrinks downward */
    var overlay = document.createElement('div');
    overlay.className = 'typing-overlay';
    overlay.style.cssText =
      'position:absolute; top:0; left:0; right:0; bottom:0;' +
      'background:' + bg + '; z-index:2; pointer-events:none;';
    qrispPre.appendChild(overlay);

    /* Create blinking cursor element */
    var cursor = document.createElement('div');
    cursor.className = 'typing-cursor-line';
    cursor.style.cssText =
      'position:absolute; left:0; right:0; height:2px; z-index:3;' +
      'background:#015999; top:0;';
    qrispPre.appendChild(cursor);

    var animated = false;

    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !animated) {
          animated = true;
          revealLines(overlay, cursor, lineHeight, totalLines, 0, preHeight);
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.25 });

    observer.observe(qrispPre);
  }

  function revealLines(overlay, cursor, lineHeight, totalLines, idx, preHeight) {
    if (idx > totalLines) {
      /* Done — remove overlay and fade cursor */
      overlay.style.display = 'none';
      setTimeout(function () {
        cursor.style.transition = 'opacity 0.5s ease';
        cursor.style.opacity = '0';
        setTimeout(function () { cursor.remove(); overlay.remove(); }, 500);
      }, 1200);
      return;
    }

    var revealedH = idx * lineHeight;
    overlay.style.top = revealedH + 'px';
    cursor.style.top = Math.min(revealedH, preHeight - 2) + 'px';

    setTimeout(function () {
      revealLines(overlay, cursor, lineHeight, totalLines, idx + 1, preHeight);
    }, 110);
  }

  /* ==========================================================
     INIT
     ========================================================== */
  document.addEventListener('DOMContentLoaded', function () {
    initScrollReveal();
    initCardTilt();
    initTypingAnimation();
  });
})();
