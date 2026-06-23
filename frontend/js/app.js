/* ═══════════════════════════════════════════════════════════════════════════
   MistralRAG — Frontend Application Logic
   Full-featured client for the RAG backend with premium interactions.
   ═══════════════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    // ── Configuration ─────────────────────────────────────────────────────
    const API_BASE = '';  // Same origin
    const SUPPORTED_TYPES = ['.pdf', '.docx', '.txt', '.md', '.csv', '.mp3', '.wav', '.m4a', '.flac', '.ogg'];
    const TOAST_DURATION = 4500;

    // ── DOM Elements ──────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);

    const elements = {
        // Sidebar
        sidebar: $('#sidebar'),
        sidebarOpenBtn: $('#sidebarOpenBtn'),
        sidebarCloseBtn: $('#sidebarCloseBtn'),

        // Upload
        dropzone: $('#dropzone'),
        fileInput: $('#fileInput'),
        uploadBtn: $('#uploadBtn'),
        uploadProgress: $('#uploadProgress'),
        progressFilename: $('#progressFilename'),
        progressPercent: $('#progressPercent'),
        progressFill: $('#progressFill'),
        progressStatus: $('#progressStatus'),

        // Documents
        documentsList: $('#documentsList'),
        emptyState: $('#emptyState'),
        docCountBadge: $('#docCountBadge'),
        totalDocs: $('#totalDocs'),
        totalChunks: $('#totalChunks'),

        // Chat
        chatMessages: $('#chatMessages'),
        chatInput: $('#chatInput'),
        chatInputWrapper: $('#chatInputWrapper'),
        sendBtn: $('#sendBtn'),
        inputHint: $('#inputHint'),
        inputHintText: $('#inputHintText'),
        kbdHints: $('#kbdHints'),
        welcomeCard: $('#welcomeCard'),
        clearChatBtn: $('#clearChatBtn'),

        // Status
        connectionDot: $('#connectionDot'),
        connectionLabel: $('#connectionLabel'),

        // Toast
        toastContainer: $('#toastContainer')
    };

    // ── State ─────────────────────────────────────────────────────────────
    let documents = [];
    let isProcessing = false;
    let isUploading = false;
    let messageCount = 0;

    // ═════════════════════════════════════════════════════════════════════
    //  UTILITIES
    // ═════════════════════════════════════════════════════════════════════

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const units = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(1024));
        return (bytes / Math.pow(1024, i)).toFixed(i > 0 ? 1 : 0) + ' ' + units[i];
    }

    function formatDate(dateString) {
        const date = new Date(dateString);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000);
        if (diff < 60) return 'just now';
        if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
        if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
        if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
        return date.toLocaleDateString();
    }

    function getFileTypeInfo(filename) {
        const ext = '.' + filename.split('.').pop().toLowerCase();
        const map = {
            '.pdf':  { icon: '📄', label: 'PDF',  cssClass: 'pdf',   abbr: 'PDF' },
            '.docx': { icon: '📝', label: 'DOCX', cssClass: 'docx',  abbr: 'DOC' },
            '.doc':  { icon: '📝', label: 'DOC',  cssClass: 'docx',  abbr: 'DOC' },
            '.txt':  { icon: '📃', label: 'TXT',  cssClass: 'txt',   abbr: 'TXT' },
            '.md':   { icon: '📃', label: 'MD',   cssClass: 'txt',   abbr: 'MD' },
            '.csv':  { icon: '📃', label: 'CSV',  cssClass: 'txt',   abbr: 'CSV' },
            '.mp3':  { icon: '🎵', label: 'MP3',  cssClass: 'audio', abbr: 'MP3' },
            '.wav':  { icon: '🎵', label: 'WAV',  cssClass: 'audio', abbr: 'WAV' },
            '.m4a':  { icon: '🎵', label: 'M4A',  cssClass: 'audio', abbr: 'M4A' },
            '.flac': { icon: '🎵', label: 'FLAC', cssClass: 'audio', abbr: 'FLC' },
            '.ogg':  { icon: '🎵', label: 'OGG',  cssClass: 'audio', abbr: 'OGG' },
        };
        return map[ext] || { icon: '📎', label: 'FILE', cssClass: 'txt', abbr: 'FILE' };
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    /** Minimal markdown → HTML (bold, italic, code, code blocks, lists, headings) */
    function renderMarkdown(text) {
        let html = escapeHtml(text);

        // Code blocks (``` ... ```)
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) => {
            return `<pre class="code-block"><code>${code.trim()}</code></pre>`;
        });

        // Inline code
        html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Bold
        html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

        // Italic
        html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

        // Headings (### , ## , # )
        html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
        html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');

        // Unordered lists
        html = html.replace(/^[\-\*] (.+)$/gm, '<li>$1</li>');
        html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

        // Ordered lists
        html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

        // Paragraphs (double newline)
        html = html.replace(/\n\n/g, '</p><p>');

        // Single newlines → <br>
        html = html.replace(/\n/g, '<br>');

        return `<p>${html}</p>`;
    }

    function isSupportedFile(filename) {
        const ext = '.' + filename.split('.').pop().toLowerCase();
        return SUPPORTED_TYPES.includes(ext);
    }

    // ═════════════════════════════════════════════════════════════════════
    //  TOAST NOTIFICATIONS
    // ═════════════════════════════════════════════════════════════════════

    function showToast(message, type = 'info') {
        const icons = { success: '✅', error: '❌', info: 'ℹ️', warning: '⚠️' };
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${escapeHtml(message)}</span>
        `;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('removing');
            setTimeout(() => toast.remove(), 350);
        }, TOAST_DURATION);
    }

    // ═════════════════════════════════════════════════════════════════════
    //  API COMMUNICATION
    // ═════════════════════════════════════════════════════════════════════

    async function apiRequest(url, options = {}) {
        try {
            const response = await fetch(`${API_BASE}${url}`, {
                headers: { 'Content-Type': 'application/json', ...options.headers },
                ...options,
            });

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(err.detail || `HTTP ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Cannot connect to server. Is the backend running?');
            }
            throw error;
        }
    }

    async function uploadFile(file) {
        if (isUploading) return;
        isUploading = true;

        // Show progress
        elements.uploadProgress.hidden = false;
        elements.progressFilename.textContent = file.name;
        elements.progressPercent.textContent = '0%';
        elements.progressFill.style.width = '0%';
        elements.progressFill.classList.add('active');
        elements.progressStatus.textContent = 'Uploading…';

        // Simulate progress since we can't easily track multipart upload progress
        let progress = 0;
        const stages = [
            { threshold: 30, text: 'Uploading…' },
            { threshold: 55, text: 'Extracting text…' },
            { threshold: 75, text: 'Generating embeddings…' },
            { threshold: 90, text: 'Storing in vector DB…' },
        ];

        const progressInterval = setInterval(() => {
            progress += Math.random() * 12;
            if (progress > 92) progress = 92;
            elements.progressFill.style.width = progress + '%';
            elements.progressPercent.textContent = Math.round(progress) + '%';

            // Update status text based on progress
            for (const stage of stages) {
                if (progress <= stage.threshold) {
                    elements.progressStatus.textContent = stage.text;
                    break;
                }
            }
        }, 400);

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${API_BASE}/api/upload`, {
                method: 'POST',
                body: formData,
            });

            clearInterval(progressInterval);

            if (!response.ok) {
                const err = await response.json().catch(() => ({ detail: response.statusText }));
                throw new Error(err.detail || `Upload failed: HTTP ${response.status}`);
            }

            const data = await response.json();

            // Complete progress animation
            elements.progressFill.style.width = '100%';
            elements.progressPercent.textContent = '100%';
            elements.progressFill.classList.remove('active');
            elements.progressStatus.textContent = 'Complete ✓';

            showToast(`"${file.name}" uploaded — ${data.document.num_chunks} chunks created`, 'success');

            setTimeout(() => {
                elements.uploadProgress.hidden = true;
            }, 1500);

            await loadDocuments();
            updateChatState();

        } catch (error) {
            clearInterval(progressInterval);
            elements.uploadProgress.hidden = true;
            showToast(error.message, 'error');
        } finally {
            isUploading = false;
        }
    }

    async function sendMessage(question) {
        if (isProcessing || !question.trim()) return;
        isProcessing = true;

        // Hide welcome card
        if (elements.welcomeCard) {
            elements.welcomeCard.style.display = 'none';
        }

        // Add user message
        addMessageToChat('user', question);

        // Clear input
        elements.chatInput.value = '';
        autoResizeTextarea();
        updateSendButton();

        // Show typing indicator
        const typingEl = showTypingIndicator();

        // Disable input while processing
        elements.chatInput.disabled = true;
        elements.sendBtn.disabled = true;

        try {
            const data = await apiRequest('/api/chat', {
                method: 'POST',
                body: JSON.stringify({ question }),
            });

            // Remove typing indicator
            typingEl.remove();

            // Add AI response
            addMessageToChat('assistant', data.answer, data.sources);

        } catch (error) {
            typingEl.remove();
            addMessageToChat('assistant', `⚠️ Error: ${error.message}`);
            showToast(error.message, 'error');
        } finally {
            isProcessing = false;
            elements.chatInput.disabled = false;
            elements.sendBtn.disabled = false;
            elements.chatInput.focus();
            updateSendButton();
        }
    }

    async function loadDocuments() {
        try {
            const data = await apiRequest('/api/documents');
            documents = data.documents || [];
            renderDocumentList();
            updateStats();
            updateChatState();
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    async function deleteDocument(docId, filename) {
        if (!confirm(`Delete "${filename}"? This cannot be undone.`)) return;

        try {
            await apiRequest(`/api/documents/${docId}`, { method: 'DELETE' });
            showToast(`"${filename}" deleted`, 'info');
            await loadDocuments();
            updateChatState();
        } catch (error) {
            showToast(error.message, 'error');
        }
    }

    async function checkHealth() {
        try {
            const data = await apiRequest('/api/health');
            elements.connectionDot.classList.remove('offline');
            elements.connectionDot.classList.add('online');
            elements.connectionLabel.textContent = 'Online';
            return true;
        } catch (error) {
            elements.connectionDot.classList.remove('online');
            elements.connectionDot.classList.add('offline');
            elements.connectionLabel.textContent = 'Offline';
            return false;
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    //  UI RENDERING
    // ═════════════════════════════════════════════════════════════════════

    function renderDocumentList() {
        const container = elements.documentsList;

        if (documents.length === 0) {
            container.innerHTML = `
                <div class="empty-state" id="emptyState">
                    <div class="empty-icon">
                        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round">
                            <path d="M22 19a2 2 0 01-2 2H4a2 2 0 01-2-2V5a2 2 0 012-2h5l2 3h9a2 2 0 012 2v11z"/>
                        </svg>
                    </div>
                    <p>No documents yet</p>
                    <span>Upload files to start chatting</span>
                </div>
            `;
            return;
        }

        container.innerHTML = documents.map((doc, idx) => {
            const ft = getFileTypeInfo(doc.filename);
            const uploadTime = doc.upload_date ? formatDate(doc.upload_date) : '';
            return `
                <div class="doc-card" style="animation-delay: ${idx * 0.06}s">
                    <div class="doc-type-icon ${ft.cssClass}">${ft.abbr}</div>
                    <div class="doc-info">
                        <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                        <div class="doc-meta">${doc.num_chunks} chunks · ${formatFileSize(doc.file_size)}${uploadTime ? ' · ' + uploadTime : ''}</div>
                    </div>
                    <button class="doc-delete-btn" onclick="window.__deleteDoc('${doc.document_id}', '${escapeHtml(doc.filename).replace(/'/g, "\\'")}')" title="Delete document">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
                            <path d="M3 6h18M19 6v14a2 2 0 01-2 2H7a2 2 0 01-2-2V6m3 0V4a2 2 0 012-2h4a2 2 0 012 2v2"/>
                        </svg>
                    </button>
                </div>
            `;
        }).join('');
    }

    function updateStats() {
        const totalDocs = documents.length;
        const totalChunks = documents.reduce((sum, d) => sum + (d.num_chunks || 0), 0);

        // Animate stat counters
        animateCounter(elements.totalDocs, totalDocs);
        animateCounter(elements.totalChunks, totalChunks);
        elements.docCountBadge.textContent = totalDocs;

        // Add subtle pulse to badge when updating
        elements.docCountBadge.style.transform = 'scale(1.15)';
        setTimeout(() => {
            elements.docCountBadge.style.transform = 'scale(1)';
        }, 250);
    }

    function animateCounter(element, target) {
        const current = parseInt(element.textContent) || 0;
        if (current === target) return;

        const duration = 400;
        const start = performance.now();

        function update(now) {
            const elapsed = now - start;
            const progress = Math.min(elapsed / duration, 1);
            // Ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            element.textContent = Math.round(current + (target - current) * eased);

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    function updateChatState() {
        const hasDocs = documents.length > 0;
        elements.chatInput.disabled = !hasDocs;
        elements.sendBtn.disabled = !hasDocs;

        if (hasDocs) {
            elements.inputHintText.textContent = '';
            elements.kbdHints.hidden = false;
        } else {
            elements.inputHintText.textContent = 'Upload documents to start chatting';
            elements.kbdHints.hidden = true;
        }
    }

    function addMessageToChat(role, content, sources = []) {
        messageCount++;
        const messageEl = document.createElement('div');
        messageEl.className = `message ${role}`;
        messageEl.style.animationDelay = '0.05s';

        const avatarLabel = role === 'user' ? 'U' : 'AI';
        let sourcesHtml = '';

        if (sources && sources.length > 0) {
            const sourceCards = sources.map(s => `
                <div class="source-card">
                    <div class="source-filename">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                        ${escapeHtml(s.filename)}
                    </div>
                    <div class="source-text">${escapeHtml(s.chunk_text)}</div>
                </div>
            `).join('');

            sourcesHtml = `
                <div class="sources-container">
                    <button class="sources-toggle" onclick="this.classList.toggle('expanded'); this.nextElementSibling.classList.toggle('visible')">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="m6 9 6 6 6-6"/></svg>
                        ${sources.length} source${sources.length > 1 ? 's' : ''} referenced
                    </button>
                    <div class="sources-list">${sourceCards}</div>
                </div>
            `;
        }

        messageEl.innerHTML = `
            <div class="message-avatar">${avatarLabel}</div>
            <div class="message-body">
                <div class="message-content">${role === 'user' ? escapeHtml(content) : renderMarkdown(content)}</div>
                ${sourcesHtml}
            </div>
        `;

        elements.chatMessages.appendChild(messageEl);
        scrollToBottom();
    }

    function showTypingIndicator() {
        const el = document.createElement('div');
        el.className = 'message assistant typing-indicator';
        el.innerHTML = `
            <div class="message-avatar">AI</div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        `;
        elements.chatMessages.appendChild(el);
        scrollToBottom();
        return el;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            elements.chatMessages.scrollTo({
                top: elements.chatMessages.scrollHeight,
                behavior: 'smooth',
            });
        });
    }

    function clearChat() {
        // Keep welcome card, remove messages
        const messages = elements.chatMessages.querySelectorAll('.message');
        if (messages.length === 0) return;

        messages.forEach((msg, i) => {
            msg.style.transition = `opacity 0.2s ease ${i * 0.03}s, transform 0.2s ease ${i * 0.03}s`;
            msg.style.opacity = '0';
            msg.style.transform = 'translateY(-10px)';
        });

        setTimeout(() => {
            messages.forEach(msg => msg.remove());
            messageCount = 0;

            // Show welcome card again
            if (elements.welcomeCard) {
                elements.welcomeCard.style.display = '';
            }
        }, 300 + messages.length * 30);

        showToast('Chat cleared', 'info');
    }

    // ═════════════════════════════════════════════════════════════════════
    //  EVENT HANDLERS
    // ═════════════════════════════════════════════════════════════════════

    // ── Textarea auto-resize ──────────────────────────────────────────────
    function autoResizeTextarea() {
        const el = elements.chatInput;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 140) + 'px';
    }

    function updateSendButton() {
        const hasText = elements.chatInput.value.trim().length > 0;
        const hasDocs = documents.length > 0;
        elements.sendBtn.disabled = !hasText || !hasDocs || isProcessing;
    }

    // ── Chat Input Events ─────────────────────────────────────────────────
    elements.chatInput.addEventListener('input', () => {
        autoResizeTextarea();
        updateSendButton();
    });

    elements.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            const question = elements.chatInput.value.trim();
            if (question && !isProcessing && documents.length > 0) {
                sendMessage(question);
            }
        }
    });

    elements.sendBtn.addEventListener('click', () => {
        const question = elements.chatInput.value.trim();
        if (question) sendMessage(question);
    });

    // ── Clear Chat ────────────────────────────────────────────────────────
    elements.clearChatBtn.addEventListener('click', clearChat);

    // ── File Upload Events ────────────────────────────────────────────────
    elements.uploadBtn.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.dropzone.addEventListener('click', () => {
        elements.fileInput.click();
    });

    elements.fileInput.addEventListener('change', (e) => {
        const files = Array.from(e.target.files);
        handleFiles(files);
        e.target.value = ''; // Reset so same file can be uploaded again
    });

    // Drag and drop
    ['dragenter', 'dragover'].forEach(evt => {
        elements.dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropzone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(evt => {
        elements.dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            elements.dropzone.classList.remove('dragover');
        });
    });

    elements.dropzone.addEventListener('drop', (e) => {
        const files = Array.from(e.dataTransfer.files);
        handleFiles(files);
    });

    // Also handle drag on the entire window for convenience
    document.addEventListener('dragover', (e) => e.preventDefault());
    document.addEventListener('drop', (e) => e.preventDefault());

    function handleFiles(files) {
        for (const file of files) {
            if (!isSupportedFile(file.name)) {
                showToast(`"${file.name}" is not a supported file type`, 'error');
                continue;
            }
            uploadFile(file);
        }
    }

    // ── Sidebar Toggle (mobile) ───────────────────────────────────────────
    let overlay = null;

    function openSidebar() {
        elements.sidebar.classList.add('open');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'sidebar-overlay visible';
            overlay.addEventListener('click', closeSidebar);
            document.body.appendChild(overlay);
        } else {
            overlay.classList.add('visible');
        }
    }

    function closeSidebar() {
        elements.sidebar.classList.remove('open');
        if (overlay) {
            overlay.classList.remove('visible');
        }
    }

    elements.sidebarOpenBtn.addEventListener('click', openSidebar);
    elements.sidebarCloseBtn.addEventListener('click', closeSidebar);

    // ── Keyboard Shortcut: Ctrl/Cmd + K to focus chat input ──────────────
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            elements.chatInput.focus();
        }
    });

    // ── Expose delete function globally ───────────────────────────────────
    window.__deleteDoc = deleteDocument;

    // ═════════════════════════════════════════════════════════════════════
    //  INITIALIZATION
    // ═════════════════════════════════════════════════════════════════════

    async function init() {
        // Initialize particle background
        initParticleBackground();

        // Check health
        const healthy = await checkHealth();
        if (!healthy) {
            showToast('Cannot connect to server. Make sure the backend is running on port 8000.', 'error');
        }

        // Load documents
        await loadDocuments();

        // Focus input if docs exist
        if (documents.length > 0) {
            elements.chatInput.focus();
        }

        // Periodic health check
        setInterval(checkHealth, 30000);

        // Add staggered feature card animations
        const featureCards = $$('.feature-card');
        featureCards.forEach((card, i) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(12px)';
            setTimeout(() => {
                card.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, 200 + i * 80);
        });
    }

    init();

})();
