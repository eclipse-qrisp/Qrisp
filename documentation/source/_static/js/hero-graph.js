(function() {
    const canvas = document.getElementById('hero-graph-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const hero = document.getElementById('hero');

    function resize() {
        const rect = hero.getBoundingClientRect();
        canvas.width = rect.width * window.devicePixelRatio;
        canvas.height = rect.height * window.devicePixelRatio;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.setTransform(window.devicePixelRatio, 0, 0, window.devicePixelRatio, 0, 0);
    }

    // Generate random nodes
    const NUM_NODES = 80;
    const CONNECTION_DIST = 180;
    let nodes = [];

    function initNodes() {
        const w = hero.offsetWidth;
        const h = hero.offsetHeight;
        nodes = [];
        for (let i = 0; i < NUM_NODES; i++) {
            nodes.push({
                x: Math.random() * w,
                y: Math.random() * h,
                vx: (Math.random() - 0.5) * 0.2,
                vy: (Math.random() - 0.5) * 0.2,
                r: 1 + Math.random() * 1.5,
                brightness: 0.15 + Math.random() * 0.25
            });
        }
    }

    function draw() {
        const w = hero.offsetWidth;
        const h = hero.offsetHeight;
        ctx.clearRect(0, 0, w, h);

        // Update positions
        for (const n of nodes) {
            n.x += n.vx;
            n.y += n.vy;
            if (n.x < 0 || n.x > w) n.vx *= -1;
            if (n.y < 0 || n.y > h) n.vy *= -1;
            n.x = Math.max(0, Math.min(w, n.x));
            n.y = Math.max(0, Math.min(h, n.y));
        }

        // Draw connections
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const dx = nodes[i].x - nodes[j].x;
                const dy = nodes[i].y - nodes[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < CONNECTION_DIST) {
                    const alpha = (1 - dist / CONNECTION_DIST) * 0.25;
                    ctx.beginPath();
                    ctx.moveTo(nodes[i].x, nodes[i].y);
                    ctx.lineTo(nodes[j].x, nodes[j].y);
                    ctx.strokeStyle = 'rgba(120, 180, 255, ' + alpha + ')';
                    ctx.lineWidth = 1;
                    ctx.stroke();
                }
            }
        }

        // Draw nodes
        for (const n of nodes) {
            // Glow
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r * 2.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(100, 170, 255, ' + (n.brightness * 0.06) + ')';
            ctx.fill();
            // Core
            ctx.beginPath();
            ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(150, 200, 255, ' + n.brightness + ')';
            ctx.fill();
        }

        requestAnimationFrame(draw);
    }

    resize();
    initNodes();
    draw();
    window.addEventListener('resize', function() { resize(); initNodes(); });
})();
