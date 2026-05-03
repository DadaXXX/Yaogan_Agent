const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sid = Math.random().toString(36).slice(2);

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Add a message row to the chat
function addMessage(role, text, images) {
    const row = document.createElement('div');
    row.className = 'msg-row ' + (role === 'you' ? 'you' : 'assistant');

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = role === 'you' ? '👤' : '🌏';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = text; // Safe: uses textContent, not innerHTML

    row.appendChild(avatar);
    row.appendChild(bubble);

    // Add images if present
    if (images && images.length > 0) {
        const imgContainer = document.createElement('div');
        imgContainer.className = 'msg-images';
        images.forEach(function(img) {
            const imgEl = document.createElement('img');
            imgEl.src = img.url;
            imgEl.alt = img.label || '';
            imgEl.title = img.label || '';
            imgEl.onclick = function() { window.open(img.url, '_blank'); };
            imgContainer.appendChild(imgEl);
        });
        row.appendChild(imgContainer);
    }

    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return row;
}

async function send(e) {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;

    // Add user message (safe: textContent)
    addMessage('you', msg);
    input.value = '';

    // Add loading indicator
    const loadRow = addMessage('assistant', '分析中...');

    try {
        const res = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg, session_id: sid })
        });
        const data = await res.json();

        // Replace loading message with actual response
        const bubble = loadRow.querySelector('.bubble');
        bubble.textContent = data.reply;

        // Add images if returned
        if (data.images && data.images.length > 0) {
            const imgContainer = document.createElement('div');
            imgContainer.className = 'msg-images';
            data.images.forEach(function(img) {
                const imgEl = document.createElement('img');
                imgEl.src = img.url;
                imgEl.alt = img.label || '';
                imgEl.title = img.label || '';
                imgEl.onclick = function() { window.open(img.url, '_blank'); };
                imgContainer.appendChild(imgEl);
            });
            loadRow.appendChild(imgContainer);
        }
    } catch (err) {
        const bubble = loadRow.querySelector('.bubble');
        bubble.textContent = '网络出错了，检查一下服务是否在运行。';
    }

    chat.scrollTop = chat.scrollHeight;
}

// Load tool tags
fetch('/api/tools')
    .then(function(r) { return r.json(); })
    .then(function(d) {
        const tags = document.getElementById('tags');
        tags.innerHTML = '';
        d.tools.forEach(function(t) {
            const span = document.createElement('span');
            span.className = 'tag';
            span.textContent = t;
            tags.appendChild(span);
        });
    });
