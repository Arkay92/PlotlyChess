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
        document.querySelectorAll('.move-ghost, .drag-ghost').forEach(el => el.remove());
        document.querySelectorAll('.drag-source').forEach(el => el.classList.remove('drag-source'));
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

    function animateMoveGhost(fromSq, toSq) {
        if (!fromSq || !toSq || fromSq === toSq) return;
        const piece = fromSq.querySelector('.piece');
        if (!piece) return;

        document.querySelectorAll('.move-ghost, .drag-ghost').forEach(el => el.remove());

        const rFrom = fromSq.getBoundingClientRect();
        const rTo = toSq.getBoundingClientRect();

        const ghost = piece.cloneNode(true);
        ghost.className = 'move-ghost';
        ghost.style.width = `${rFrom.width * 0.86}px`;
        ghost.style.left = `${rFrom.left + rFrom.width / 2}px`;
        ghost.style.top = `${rFrom.top + rFrom.height / 2}px`;
        document.body.appendChild(ghost);

        fromSq.classList.add('drag-source');
        const targetPiece = toSq.querySelector('.piece');
        if (targetPiece) targetPiece.style.opacity = '0';

        document.querySelectorAll('.square.selected, .square.legal-target, .square.legal-capture, .drag-legal, .drag-over')
            .forEach(el => el.classList.remove('selected', 'legal-target', 'legal-capture', 'drag-legal', 'drag-over'));

        requestAnimationFrame(() => {
            ghost.style.transition = 'left 0.14s cubic-bezier(0.2, 0, 0.2, 1), top 0.14s cubic-bezier(0.2, 0, 0.2, 1)';
            ghost.style.left = `${rTo.left + rTo.width / 2}px`;
            ghost.style.top = `${rTo.top + rTo.height / 2}px`;
        });

        setTimeout(() => ghost.remove(), 250);
    }

    function cleanup() {
        if (gesture?.ghost) gesture.ghost.remove();
        gesture = null;
        document.querySelectorAll('.drag-over,.drag-legal').forEach(el => el.classList.remove('drag-over', 'drag-legal'));
    }

    document.addEventListener('pointerdown', (event) => {
        const source = squareAt(event.target);
        if (!source || event.button !== 0 || isBusy() || gesture || !event.isPrimary) return;
        const canDrag = source.dataset.movable === 'true';
        gesture = {
            source,
            id: event.pointerId,
            x: event.clientX,
            y: event.clientY,
            dragging: false,
            canDrag: canDrag
        };
        try {
            source.setPointerCapture(event.pointerId);
        } catch (_) {}
    });

    document.addEventListener('pointermove', (event) => {
        if (!gesture || gesture.id !== event.pointerId || gesture.cancelled) return;
        if (!gesture.canDrag) return;
        if (!gesture.dragging && Math.hypot(event.clientX - gesture.x, event.clientY - gesture.y) > 6) {
            gesture.dragging = true;
            gesture.source.classList.add('drag-source');
            const piece = gesture.source.querySelector('.piece');
            if (piece) {
                const ghost = piece.cloneNode(true);
                const rect = gesture.source.getBoundingClientRect();
                ghost.className = 'drag-ghost';
                ghost.style.width = `${rect.width * .86}px`;
                document.body.appendChild(ghost);
                gesture.ghost = ghost;
            }
            const legal = (gesture.source.dataset.legal || '').split(',').filter(Boolean);
            grid()?.querySelectorAll('.square').forEach(el => {
                if (legal.includes(el.dataset.square)) el.classList.add('drag-legal');
            });
        }
        if (!gesture.dragging || !gesture.ghost) return;
        event.preventDefault();
        gesture.ghost.style.left = `${event.clientX}px`;
        gesture.ghost.style.top = `${event.clientY}px`;
        document.querySelector('.drag-over')?.classList.remove('drag-over');
        squareAt(document.elementFromPoint(event.clientX, event.clientY))?.classList.add('drag-over');
    }, {passive: false});

    document.addEventListener('pointerup', (event) => {
        if (!gesture || gesture.id !== event.pointerId) return;
        const {source, dragging, x, y} = gesture;
        try {
            if (source.hasPointerCapture(event.pointerId)) source.releasePointerCapture(event.pointerId);
        } catch (_) {}
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

            const legalMoves = (source.dataset.legal || '').split(',').filter(Boolean);
            const isLegal = target && legalMoves.includes(target.dataset.square);

            if (target && target !== source && isLegal) {
                if (gesture.ghost) {
                    const rTo = target.getBoundingClientRect();
                    const ghost = gesture.ghost;
                    gesture.ghost = null;
                    ghost.className = 'move-ghost';
                    ghost.style.transition = 'left 0.12s cubic-bezier(0.2, 0, 0.2, 1), top 0.12s cubic-bezier(0.2, 0, 0.2, 1)';
                    ghost.style.left = `${rTo.left + rTo.width / 2}px`;
                    ghost.style.top = `${rTo.top + rTo.height / 2}px`;
                    setTimeout(() => ghost.remove(), 250);
                }
                cleanup();
                send({type: 'drop', from: Number(source.dataset.square), to: Number(target.dataset.square)});
            } else {
                if (gesture.ghost) {
                    const rFrom = source.getBoundingClientRect();
                    const ghost = gesture.ghost;
                    gesture.ghost = null;
                    ghost.className = 'move-ghost';
                    ghost.style.transition = 'left 0.12s cubic-bezier(0.2, 0, 0.2, 1), top 0.12s cubic-bezier(0.2, 0, 0.2, 1)';
                    ghost.style.left = `${rFrom.left + rFrom.width / 2}px`;
                    ghost.style.top = `${rFrom.top + rFrom.height / 2}px`;
                    setTimeout(() => {
                        ghost.remove();
                        source.classList.remove('drag-source');
                    }, 140);
                }
                cleanup();
            }
        } else {
            const dist = Math.hypot(event.clientX - x, event.clientY - y);
            cleanup();
            if (dist < 12) {
                suppressClickUntil = performance.now() + 400;
                suppressedSquare = source.dataset.square;

                const selectedSq = document.querySelector('.square.selected');
                if (selectedSq && selectedSq !== source && (source.classList.contains('legal-target') || source.classList.contains('legal-capture'))) {
                    animateMoveGhost(selectedSq, source);
                }

                send({type: 'click', square: Number(source.dataset.square)});
            }
        }
    });

    document.addEventListener('click', (event) => {
        const target = event.target instanceof Element ? event.target : null;
        if (!target) return;
        const friendsToggle = target.closest('#friends-toggle-btn, #friends-close-btn');
        if (friendsToggle) {
            const box = document.getElementById('friends-widget');
            if (box) {
                box.classList.toggle('friends-open');
            }
            return;
        }
        const profileBtn = target.closest('[data-view-profile]');
        if (profileBtn) {
            const user = profileBtn.dataset.viewProfile || '';
            if (user) {
                send({type: 'view-profile', username: user});
            }
            return;
        }
        const toggleFriendBtn = target.closest('[data-toggle-friend]');
        if (toggleFriendBtn) {
            const friendUser = toggleFriendBtn.dataset.toggleFriend || '';
            if (friendUser) {
                send({type: 'toggle-friend', username: friendUser});
            }
            return;
        }
        const burgerBtn = target.closest('#burger-menu-btn');
        const siteNav = document.getElementById('site-nav');
        if (burgerBtn) {
            const isOpen = siteNav?.classList.toggle('nav-open');
            burgerBtn.setAttribute('aria-expanded', String(Boolean(isOpen)));
            return;
        }
        if (siteNav?.classList.contains('nav-open')) {
            if (!siteNav.contains(target) || target.closest('#site-nav button, #site-nav a')) {
                siteNav.classList.remove('nav-open');
                document.getElementById('burger-menu-btn')?.setAttribute('aria-expanded', 'false');
            }
        }
        const userMenu = document.getElementById('user-menu');
        if (userMenu?.hasAttribute('open') && (!userMenu.contains(target) || target.closest('.user-dropdown-menu button'))) {
            setTimeout(() => userMenu.removeAttribute('open'), 50);
        }
        const homeOrLeaveBtn = target.closest('#brand-home-btn, #nav-play-btn, #mp-leave-btn, #nav-profile-btn, #nav-leaderboard-btn, #mp-tab-lobby-btn, #mp-tab-find-btn, #mp-choose-join-btn, #nav-about-btn, #footer-about-btn, #menu-about-btn, #about-back-btn');
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
        if (event.key === 'Escape') {
            const siteNav = document.getElementById('site-nav');
            if (siteNav?.classList.contains('nav-open')) {
                siteNav.classList.remove('nav-open');
                document.getElementById('burger-menu-btn')?.setAttribute('aria-expanded', 'false');
            }
        }
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
