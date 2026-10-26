/* Pointer capture, keyboard controls and live multiplayer WebSocket bridge. */
(() => {
    let gesture = null;
    let suppressClickUntil = 0;
    let suppressedSquare = null;
    let busy = false;
    let busyTimer = null;
    let queuedAction = null;
    let sequence = 0;
    let socket = null;
    let socketCode = null;
    let reconnectTimer = null;
    let wsSupported = true;
    let suppressSocketUntil = 0;

    const squareAt = (target) => target instanceof Element ? target.closest('.square') : null;
    const grid = () => document.querySelector('.chess-grid');
    const isBusy = () => busy || document.querySelector('#board[data-busy="true"]');

    function clearBusy() {
        busy = false;
        if (busyTimer) {
            clearTimeout(busyTimer);
            busyTimer = null;
        }
        document.getElementById('board')?.setAttribute('aria-busy', 'false');
        if (queuedAction) {
            const next = queuedAction;
            queuedAction = null;
            send(next);
        }
    }

    function send(action) {
        console.log('[BOARD.JS] send called, type:', action?.type, 'has_set_props:', Boolean(window.dash_clientside?.set_props));
        if (!window.dash_clientside?.set_props || !action) return;
        const isBoardMove = action.type === 'click' || action.type === 'drop' || action.type === 'mp-sync';
        if (isBoardMove && isBusy()) {
            if (action.type !== 'mp-sync') queuedAction = action;
            return;
        }
        if (!isBoardMove) {
            queuedAction = null;
        }
        busy = true;
        if (busyTimer) clearTimeout(busyTimer);
        busyTimer = setTimeout(clearBusy, 400);
        document.getElementById('board')?.setAttribute('aria-busy', 'true');
        window.dash_clientside.set_props('board-action', {data: {...action, nonce: ++sequence}});
    }

    function applySyncEvent(msg, code) {
        if (performance.now() < suppressSocketUntil) return;
        if (!msg || msg.type !== 'sync' || msg.code !== code) return;
        const activeSession = document.querySelector('.mp-session');
        if (!activeSession || activeSession.offsetParent === null || activeSession.dataset.roomCode !== code) return;
        const currentVersion = Number(activeSession.dataset.mpVersion || 0);
        const currentConnected = activeSession.dataset.mpConnected === 'true';
        if (Number(msg.version) > currentVersion || Boolean(msg.connected) !== currentConnected) {
            send({type: 'mp-sync', version: msg.version});
        }
    }

    function closeSocket() {
        if (reconnectTimer) {
            clearTimeout(reconnectTimer);
            reconnectTimer = null;
        }
        if (socket) {
            const current = socket;
            socket = null;
            socketCode = null;
            current.onclose = null;
            current.onmessage = null;
            try {
                current.close();
            } catch (_) {}
        }
    }

    function syncMultiplayerSocket() {
        if (performance.now() < suppressSocketUntil) {
            closeSocket();
            return;
        }
        const session = document.querySelector('.mp-session');
        const roomVisible = Boolean(session && session.offsetParent !== null);
        const code = roomVisible ? (session.dataset.roomCode || null) : null;
        if (code) {
            const expectedSearch = `?code=${encodeURIComponent(code)}`;
            if (window.location.search !== expectedSearch && window.history?.replaceState) {
                window.history.replaceState(null, '', `${window.location.pathname}${expectedSearch}`);
            }
        }
        if (!wsSupported) return;
        if (!code) {
            closeSocket();
            return;
        }
        if (socket && socketCode === code && socket.readyState <= WebSocket.OPEN) {
            return;
        }
        closeSocket();
        socketCode = code;
        let receivedFrame = false;
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws?code=${encodeURIComponent(code)}`;
        try {
            socket = new WebSocket(wsUrl);
        } catch (_) {
            socket = null;
            wsSupported = false;
        }
        if (socket) {
            socket.onmessage = (event) => {
                receivedFrame = true;
                try {
                    applySyncEvent(JSON.parse(event.data), code);
                } catch (_) {}
            };
            socket.onclose = () => {
                socket = null;
                if (!receivedFrame) {
                    // Server is running Gunicorn sync workers (e.g. Plotly Cloud); rely on #mp-poll dcc.Interval
                    wsSupported = false;
                    return;
                }
                if (document.querySelector('.mp-session')?.dataset.roomCode === code) {
                    reconnectTimer = setTimeout(syncMultiplayerSocket, 2000);
                }
            };
        }
    }

    function cleanup() {
        if (gesture?.ghost) gesture.ghost.remove();
        gesture?.source.classList.remove('drag-source');
        document.querySelectorAll('.drag-over,.drag-legal').forEach(el => el.classList.remove('drag-over', 'drag-legal'));
        gesture = null;
    }

    document.addEventListener('pointerdown', (event) => {
        const source = squareAt(event.target);
        if (!source || event.button !== 0 || isBusy() || gesture || !event.isPrimary) return;
        if (source.dataset.movable !== 'true') return;
        gesture = {source, id: event.pointerId, x: event.clientX, y: event.clientY, dragging: false};
        source.setPointerCapture(event.pointerId);
    });

    document.addEventListener('pointermove', (event) => {
        if (!gesture || gesture.id !== event.pointerId || gesture.cancelled) return;
        if (!gesture.dragging && Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) > 6) {
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
        squareAt(document.elementFromPoint(event.clientX, event.clientY))?.classList.add('drag-over');
    }, {passive: false});

    document.addEventListener('pointerup', (event) => {
        if (!gesture || gesture.id !== event.pointerId) return;
        const {source, dragging} = gesture;
        if (source.hasPointerCapture(event.pointerId)) source.releasePointerCapture(event.pointerId);
        if (gesture.cancelled) {
            suppressClickUntil = performance.now() + 400;
            suppressedSquare = source.dataset.square;
            cleanup();
            return;
        }
        if (dragging) {
            const target = squareAt(document.elementFromPoint(event.clientX, event.clientY));
            suppressClickUntil = performance.now() + 400;
            suppressedSquare = source.dataset.square;
            cleanup();
            if (target && target !== source) send({type: 'drop', from: Number(source.dataset.square), to: Number(target.dataset.square)});
        } else {
            cleanup();
            suppressClickUntil = performance.now() + 400;
            suppressedSquare = source.dataset.square;
            send({type: 'click', square: Number(source.dataset.square)});
        }
    });

    document.addEventListener('click', (event) => {
        const target = event.target instanceof Element ? event.target : null;
        if (!target) return;
        const userMenu = document.getElementById('user-menu');
        if (userMenu?.hasAttribute('open') && (!userMenu.contains(target) || target.closest('.user-dropdown-menu button'))) {
            setTimeout(() => userMenu.removeAttribute('open'), 50);
        }
        const homeOrLeaveBtn = target.closest('#brand-home-btn, #nav-play-btn, #mp-leave-btn, #nav-profile-btn, #nav-leaderboard-btn, #mp-tab-lobby-btn, #mp-tab-find-btn, #mp-choose-join-btn');
        if (homeOrLeaveBtn) {
            queuedAction = null;
            suppressSocketUntil = performance.now() + 1500;
            document.querySelector('.mp-session')?.removeAttribute('data-room-code');
            closeSocket();
            if (window.location.search && window.history?.replaceState) {
                window.history.replaceState(null, '', window.location.pathname);
            }
            return;
        }
        const resumeBtn = target.closest('[data-resume-game]');
        if (resumeBtn) {
            const gid = resumeBtn.dataset.resumeGame || '';
            console.log('[BOARD.JS] resumeBtn clicked, gid:', gid);
            if (gid.startsWith('mp:')) {
                suppressSocketUntil = 0;
                const code = gid.slice(3);
                if (code && window.history?.replaceState) {
                    window.history.replaceState(null, '', `${window.location.pathname}?code=${encodeURIComponent(code)}`);
                }
            } else {
                suppressSocketUntil = performance.now() + 1500;
                document.querySelector('.mp-session')?.removeAttribute('data-room-code');
                closeSocket();
                if (window.location.search && window.history?.replaceState) {
                    window.history.replaceState(null, '', window.location.pathname);
                }
            }
            send({type: 'resume-game', game_id: gid});
            return;
        }
        const dismissBtn = target.closest('[data-dismiss-game]');
        if (dismissBtn) {
            send({type: 'dismiss-game', game_id: dismissBtn.dataset.dismissGame});
            return;
        }
        const joinOpenBtn = target.closest('[data-join-open-game]');
        if (joinOpenBtn) {
            const code = joinOpenBtn.dataset.joinOpenGame || '';
            suppressSocketUntil = 0;
            if (code && window.history?.replaceState) {
                window.history.replaceState(null, '', `${window.location.pathname}?code=${encodeURIComponent(code)}`);
            }
            send({type: 'mp-join', code});
            return;
        }
        const copyBtn = target.closest('#mp-copy-btn');
        if (copyBtn) {
            const code = copyBtn.dataset.code || '';
            if (code && navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(code).catch(() => {});
            }
            copyBtn.textContent = 'Copied!';
            setTimeout(() => { copyBtn.textContent = 'Copy code'; }, 1400);
            return;
        }
        const linkBtn = target.closest('#mp-copy-link-btn');
        if (linkBtn) {
            const code = linkBtn.dataset.code || '';
            const link = `${window.location.origin}${window.location.pathname}?code=${encodeURIComponent(code)}`;
            if (code && navigator.clipboard?.writeText) {
                navigator.clipboard.writeText(link).catch(() => {});
            }
            linkBtn.textContent = 'Link copied!';
            setTimeout(() => { linkBtn.textContent = 'Copy link'; }, 1400);
            return;
        }
        const square = squareAt(target);
        if (!square) return;
        if (event.detail !== 0 && performance.now() < suppressClickUntil && square.dataset.square === suppressedSquare) {
            suppressClickUntil = 0;
            return;
        }
        send({type: 'click', square: Number(square.dataset.square)});
    });

    document.addEventListener('pointercancel', cleanup);

    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && gesture) {
            gesture.cancelled = true;
            gesture.ghost?.remove();
            gesture.source.classList.remove('drag-source');
            document.querySelectorAll('.drag-over,.drag-legal').forEach(el => el.classList.remove('drag-over', 'drag-legal'));
        }
        const square = squareAt(event.target);
        if (!square || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
        event.preventDefault();
        const squares = [...grid().querySelectorAll('.square')];
        const offset = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -8, ArrowDown: 8}[event.key];
        squares[squares.indexOf(square) + offset]?.focus();
    });

    window.addEventListener('blur', cleanup);

    const observer = new MutationObserver((records) => {
        if (records.some(record => record.target?.id === 'board' || record.target?.classList?.contains('chess-grid'))) {
            clearBusy();
        }
        syncMultiplayerSocket();
    });
    observer.observe(document.documentElement, {attributes: true, childList: true, attributeFilter: ['data-busy', 'data-room-code', 'data-mp-version'], subtree: true});
})();
