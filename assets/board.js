/* Pointer capture supports mouse, pen and touch. Python validates all moves. */
(() => {
    let gesture = null;
    let suppressClickUntil = 0;
    let suppressedSquare = null;
    let busy = false;
    let queuedAction = null;
    let sequence = 0;
    const squareAt = (target) => target instanceof Element ? target.closest('.square') : null;
    const grid = () => document.querySelector('.chess-grid');
    const isBusy = () => busy || document.querySelector('#board[data-busy="true"]');
    function send(action) {
        if (!window.dash_clientside?.set_props) return;
        if (isBusy()) {
            queuedAction = action;
            return;
        }
        busy = true;
        document.getElementById('board')?.setAttribute('aria-busy', 'true');
        window.dash_clientside.set_props('board-action', {data: {...action, nonce: ++sequence}});
    }
    function cleanup() {
        if (gesture?.ghost) gesture.ghost.remove();
        gesture?.source.classList.remove('drag-source');
        document.querySelectorAll('.drag-over,.drag-legal').forEach(el => el.classList.remove('drag-over','drag-legal'));
        gesture = null;
    }
    document.addEventListener('pointerdown', (event) => {
        const source = squareAt(event.target);
        if (!source || event.button !== 0 || isBusy() || gesture || !event.isPrimary) return;
        if (source.dataset.movable !== 'true') return;
        gesture = {source, id:event.pointerId, x:event.clientX, y:event.clientY, dragging:false};
        source.setPointerCapture(event.pointerId);
    });
    document.addEventListener('pointermove', (event) => {
        if (!gesture || gesture.id !== event.pointerId || gesture.cancelled) return;
        if (!gesture.dragging && Math.hypot(event.clientX-gesture.x,event.clientY-gesture.y) > 6) {
            gesture.dragging = true;
            gesture.source.classList.add('drag-source');
            const ghost = gesture.source.querySelector('.piece').cloneNode(true);
            const rect = gesture.source.getBoundingClientRect();
            ghost.className = 'drag-ghost';
            ghost.style.width = `${rect.width * .86}px`;
            document.body.appendChild(ghost);
            gesture.ghost = ghost;
            const legal = gesture.source.dataset.legal.split(',');
            grid().querySelectorAll('.square').forEach(el => {
                if (legal.includes(el.dataset.square)) el.classList.add('drag-legal');
            });
        }
        if (!gesture.dragging) return;
        event.preventDefault();
        gesture.ghost.style.left = `${event.clientX}px`;
        gesture.ghost.style.top = `${event.clientY}px`;
        document.querySelector('.drag-over')?.classList.remove('drag-over');
        squareAt(document.elementFromPoint(event.clientX,event.clientY))?.classList.add('drag-over');
    }, {passive:false});
    document.addEventListener('pointerup', (event) => {
        if (!gesture || gesture.id !== event.pointerId) return;
        const {source, dragging} = gesture;
        if (source.hasPointerCapture(event.pointerId)) source.releasePointerCapture(event.pointerId);
        if (gesture.cancelled) {
            suppressClickUntil = performance.now()+400;
            suppressedSquare = source.dataset.square;
            cleanup();
            return;
        }
        if (dragging) {
            const target = squareAt(document.elementFromPoint(event.clientX,event.clientY));
            suppressClickUntil = performance.now()+400;
            suppressedSquare = source.dataset.square;
            cleanup();
            if (target && target !== source) send({type:'drop',from:Number(source.dataset.square),to:Number(target.dataset.square)});
        } else {
            cleanup();
            // Pointer capture may retarget the click; explicitly select the source.
            suppressClickUntil = performance.now()+400;
            suppressedSquare = source.dataset.square;
            send({type:'click',square:Number(source.dataset.square)});
        }
    });
    document.addEventListener('click', (event) => {
        const square = squareAt(event.target);
        if (!square) return;
        if (event.detail !== 0 && performance.now() < suppressClickUntil && square.dataset.square === suppressedSquare) {
            suppressClickUntil = 0;
            return;
        }
        send({type:'click',square:Number(square.dataset.square)});
    });
    document.addEventListener('pointercancel', cleanup);
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && gesture) {
            gesture.cancelled = true;
            gesture.ghost?.remove();
            gesture.source.classList.remove('drag-source');
            document.querySelectorAll('.drag-over,.drag-legal').forEach(el => el.classList.remove('drag-over','drag-legal'));
        }
        const square = squareAt(event.target);
        if (!square || !['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(event.key)) return;
        event.preventDefault();
        const squares = [...grid().querySelectorAll('.square')];
        const offset = {ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8}[event.key];
        squares[squares.indexOf(square)+offset]?.focus();
    });
    window.addEventListener('blur', cleanup);
    // Dash replaces the grid after every action, including illegal drops.
    const observer = new MutationObserver((records) => {
        if (records.some(record => record.type === 'attributes' &&
                record.target.id === 'board' && record.target.dataset.busy === 'false')) {
            busy = false;
            document.getElementById('board')?.setAttribute('aria-busy','false');
            if (queuedAction) {
                const next = queuedAction;
                queuedAction = null;
                send(next);
            }
        }
    });
    observer.observe(document.documentElement,{attributes:true,attributeFilter:['data-busy'],subtree:true});
})();
