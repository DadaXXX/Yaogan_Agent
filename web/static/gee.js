// ═══ GEE 数据下载面板 ═══

var geeMap = null;
var geeMarker = null;
var geeShpLayer = null;
var geeDrawnRect = null;
var geeDrawControl = null;
var geeFootprintLayer = null;
var geeTileLayer = null;
var geeSearchResults = [];
var geeAbortController = null;
var geeFootprintsVisible = true;

// ── 初始化 Leaflet 地图 ────────────────────────────────

function initGeeMap() {
    if (geeMap) return;

    geeMap = L.map('gee-map').setView([35, 105], 4);

    // 底图图层
    var osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap',
        maxZoom: 18,
    });

    var satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri',
        maxZoom: 18,
    });

    osmLayer.addTo(geeMap);

    // 图层切换控件
    L.control.layers({
        "OpenStreetMap": osmLayer,
        "卫星影像": satelliteLayer,
    }).addTo(geeMap);

    // 绘制控件 — 支持矩形和多边形
    var drawnItems = new L.FeatureGroup();
    geeMap.addLayer(drawnItems);

    geeDrawControl = new L.Control.Draw({
        draw: {
            polyline: false,
            polygon: {
                allowIntersection: false,
                shapeOptions: { color: '#0071e3', weight: 2, fillOpacity: 0.1 }
            },
            rectangle: {
                shapeOptions: { color: '#0071e3', weight: 2, fillOpacity: 0.1 }
            },
            circle: false,
            circlemarker: false,
            marker: true,
        },
        edit: {
            featureGroup: drawnItems,
            remove: true,
        },
    });
    geeMap.addControl(geeDrawControl);

    // 绘制完成事件
    geeMap.on(L.Draw.Event.CREATED, function(e) {
        drawnItems.clearLayers();
        drawnItems.addLayer(e.layer);
        geeDrawnRect = e.layer;

        if (e.layerType === 'marker') {
            var latlng = e.layer.getLatLng();
            document.getElementById('gee-lat').value = latlng.lat.toFixed(4);
            document.getElementById('gee-lon').value = latlng.lng.toFixed(4);
            document.getElementById('gee-coord').textContent =
                '📍 ' + latlng.lng.toFixed(4) + ', ' + latlng.lat.toFixed(4);
            geeDrawnRect = null;
        } else {
            var bounds = e.layer.getBounds();
            document.getElementById('gee-coord').textContent =
                '📐 选区: ' + bounds.getSouthWest().lng.toFixed(3) + ',' +
                bounds.getSouthWest().lat.toFixed(3) + ' → ' +
                bounds.getNorthEast().lng.toFixed(3) + ',' +
                bounds.getNorthEast().lat.toFixed(3);
        }
    });

    geeMap.on(L.Draw.Event.DELETED, function() {
        geeDrawnRect = null;
        document.getElementById('gee-coord').textContent = '📍 点击地图选择位置 / 使用工具栏绘制矩形选区';
    });

    // 点击地图 → 选点
    geeMap.on('click', function(e) {
        if (geeDrawControl._toolbars.draw._activeMode) return; // 正在绘制时不处理

        var lat = e.latlng.lat.toFixed(4);
        var lon = e.latlng.lng.toFixed(4);

        document.getElementById('gee-lat').value = lat;
        document.getElementById('gee-lon').value = lon;
        document.getElementById('gee-coord').textContent = '📍 ' + lon + ', ' + lat;

        if (geeMarker) {
            geeMarker.setLatLng(e.latlng);
        } else {
            geeMarker = L.marker(e.latlng).addTo(geeMap);
        }
        geeDrawnRect = null;
    });

    // Fix map rendering after panel switch
    setTimeout(function() { geeMap.invalidateSize(); }, 100);

    // Check GEE auth status
    geeCheckAuth();
}

// ── 导航面板切换 ────────────────────────────────────────

document.addEventListener('DOMContentLoaded', function() {
    var navLinks = document.querySelectorAll('aside nav a');

    navLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            var panel = this.getAttribute('data-panel');
            if (!panel) return;

            // Update nav active state
            navLinks.forEach(function(l) { l.classList.remove('active'); });
            this.classList.add('active');

            // Show target panel
            document.querySelectorAll('.panel').forEach(function(p) {
                p.classList.remove('active');
            });
            var target = document.getElementById('panel-' + panel);
            if (target) {
                target.classList.add('active');
            }

            // Init map when switching to GEE panel
            if (panel === 'gee') {
                initGeeMap();
                setTimeout(function() {
                    if (geeMap) geeMap.invalidateSize();
                }, 50);
            }

            // Load output images
            if (panel === 'output') {
                loadOutputImages();
            }
        });
    });
});

// ── 搜索影像 ────────────────────────────────────────────

async function geeSearch() {
    var lon = parseFloat(document.getElementById('gee-lon').value);
    var lat = parseFloat(document.getElementById('gee-lat').value);

    // 如果有矩形选区，用几何搜索
    if (geeDrawnRect && isNaN(lon)) {
        var bounds = geeDrawnRect.getBounds();
        lon = (bounds.getWest() + bounds.getEast()) / 2;
        lat = (bounds.getSouth() + bounds.getNorth()) / 2;
        document.getElementById('gee-lon').value = lon.toFixed(4);
        document.getElementById('gee-lat').value = lat.toFixed(4);
    }

    if (isNaN(lon) || isNaN(lat)) {
        alert('请先在地图上点击选择位置，或绘制矩形选区');
        return;
    }

    var body = {
        collection: document.getElementById('gee-collection').value,
        lon: lon,
        lat: lat,
        buffer_km: parseFloat(document.getElementById('gee-buffer').value) || 20,
        start_date: document.getElementById('gee-start').value || '2020-01-01',
        end_date: document.getElementById('gee-end').value || '2024-12-31',
        cloud_cover: parseFloat(document.getElementById('gee-cloud').value) || 10,
        max_results: parseInt(document.getElementById('gee-max').value) || 50,
    };

    var searchBtn = document.getElementById('gee-search-btn');
    var cancelBtn = document.getElementById('gee-cancel-btn');
    searchBtn.disabled = true;
    searchBtn.textContent = '搜索中...';
    cancelBtn.style.display = 'inline-flex';

    geeAbortController = new AbortController();

    try {
        var resp = await fetch('/api/gee/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
            signal: geeAbortController.signal,
        });

        if (!resp.ok) {
            var err = await resp.json();
            throw new Error(err.detail || '搜索失败');
        }

        var data = await resp.json();
        geeSearchResults = data.images || [];
        renderGeeResults(data);
    } catch (e) {
        if (e.name === 'AbortError') {
            alert('搜索已取消');
        } else {
            alert('搜索失败: ' + e.message);
        }
    } finally {
        searchBtn.disabled = false;
        searchBtn.textContent = '🔍 搜索影像';
        cancelBtn.style.display = 'none';
        geeAbortController = null;
    }
}

// ── 切换范围显示 ────────────────────────────────────────

function geeToggleFootprints() {
    var btn = document.getElementById('gee-toggle-footprint');
    if (geeFootprintsVisible) {
        if (geeFootprintLayer) geeMap.removeLayer(geeFootprintLayer);
        btn.textContent = '🔲 显示范围';
        geeFootprintsVisible = false;
    } else {
        if (geeFootprintLayer) geeFootprintLayer.addTo(geeMap);
        btn.textContent = '🔲 隐藏范围';
        geeFootprintsVisible = true;
    }
}

// ── 取消搜索 ────────────────────────────────────────────

function geeCancelSearch() {
    if (geeAbortController) {
        geeAbortController.abort();
    }
}

// ── 渲染搜索结果 ────────────────────────────────────────

function renderGeeResults(data) {
    var container = document.getElementById('gee-results');
    var list = document.getElementById('gee-results-list');
    var countEl = document.getElementById('gee-results-count');

    container.style.display = 'block';
    countEl.textContent = '搜索结果 (共 ' + data.count + ' 景)';

    list.innerHTML = '';

    // Draw image footprints on map
    if (geeFootprintLayer) {
        geeMap.removeLayer(geeFootprintLayer);
    }
    geeFootprintLayer = L.featureGroup().addTo(geeMap);

    data.images.forEach(function(img, idx) {

        // Draw footprint if bounds available
        if (img.bounds && img.bounds.length > 0) {
            var latlngs = img.bounds.map(function(c) { return [c[1], c[0]]; });
            var polygon = L.polygon(latlngs, {
                color: '#0071e3',
                weight: 1,
                opacity: 0.6,
                fillOpacity: 0.08,
                dashArray: '4,3',
            });
            polygon.bindTooltip(img.id, {sticky: true, opacity: 0.8});
            polygon.on('click', function() {
                // Toggle checkbox when clicking footprint
                var cb = document.querySelector('#gee-results-list input[data-idx="' + idx + '"]');
                if (cb) { cb.checked = !cb.checked; updateSelectedCount(); }
            });
            geeFootprintLayer.addLayer(polygon);
        }
        var item = document.createElement('div');
        item.className = 'gee-result-item';
        item.setAttribute('data-idx', idx);

        var cc = img.cloud_cover;
        var ccClass = cc === null ? '' : (cc < 5 ? 'low' : (cc < 15 ? 'mid' : 'high'));
        var ccText = cc === null ? 'N/A' : (cc.toFixed(1) + '%');

        item.innerHTML =
            '<input type="checkbox" data-idx="' + idx + '" onchange="updateSelectedCount()">' +
            '<div class="gee-img-info">' +
            '<div class="gee-img-id">' + escapeHtml(img.id) + '</div>' +
            '<div class="gee-img-meta">' + (img.date || '未知日期') + '</div>' +
            '</div>' +
            '<span class="gee-img-cloud ' + ccClass + '">' + ccText + '</span>' +
            '<button class="gee-btn-sm" onclick="event.stopPropagation();geeShowOnMap(' + idx + ')" title="在地图上显示">🗺️</button>';

        item.addEventListener('click', function(e) {
            if (e.target.type === 'checkbox') return;
            var cb = this.querySelector('input[type="checkbox"]');
            cb.checked = !cb.checked;
            updateSelectedCount();
        });

        list.appendChild(item);
    });

    updateSelectedCount();

    // Fit map to show all footprints
    if (geeFootprintLayer.getLayers().length > 0) {
        geeMap.fitBounds(geeFootprintLayer.getBounds(), {padding: [20, 20]});
    }

    // Show toggle button, reset state
    document.getElementById('gee-toggle-footprint').style.display = 'inline-flex';
    geeFootprintsVisible = true;
    document.getElementById('gee-toggle-footprint').textContent = '🔲 隐藏范围';
}

function geeSelectAll() {
    document.querySelectorAll('#gee-results-list input[type="checkbox"]').forEach(function(cb) {
        cb.checked = true;
    });
    updateSelectedCount();
}

function geeSelectNone() {
    document.querySelectorAll('#gee-results-list input[type="checkbox"]').forEach(function(cb) {
        cb.checked = false;
    });
    updateSelectedCount();
}

function updateSelectedCount() {
    var checked = document.querySelectorAll('#gee-results-list input[type="checkbox"]:checked');
    document.getElementById('gee-selected-count').textContent = '已选 ' + checked.length + ' 景';

    document.querySelectorAll('.gee-result-item').forEach(function(item) {
        var cb = item.querySelector('input[type="checkbox"]');
        item.classList.toggle('selected', cb && cb.checked);
    });
}

function getSelectedImageIds() {
    var ids = [];
    document.querySelectorAll('#gee-results-list input[type="checkbox"]:checked').forEach(function(cb) {
        var idx = parseInt(cb.getAttribute('data-idx'));
        if (geeSearchResults[idx]) {
            ids.push(geeSearchResults[idx].id);
        }
    });
    return ids;
}

// ── 按周期合成下载 ──────────────────────────────────────

async function geeComposite() {
    var lon = parseFloat(document.getElementById('gee-lon').value);
    var lat = parseFloat(document.getElementById('gee-lat').value);
    if (isNaN(lon) || isNaN(lat)) {
        alert('请先选择位置');
        return;
    }

    var periodDays = parseInt(document.getElementById('gee-composite').value);
    if (periodDays === 0) {
        alert('请先选择合成周期（如每16天）');
        return;
    }

    var body = {
        collection: document.getElementById('gee-collection').value,
        lon: lon,
        lat: lat,
        buffer_km: parseFloat(document.getElementById('gee-buffer').value) || 20,
        start_date: document.getElementById('gee-start').value || '2024-01-01',
        end_date: document.getElementById('gee-end').value || '2024-12-31',
        cloud_cover: parseFloat(document.getElementById('gee-cloud').value) || 10,
        period_days: periodDays,
        bands: document.getElementById('gee-bands').value.trim(),
        scale: document.getElementById('gee-scale').value || '10',
        cloud_mask: document.getElementById('gee-cloud-mask').checked,
        add_ndvi: document.getElementById('gee-add-ndvi').checked,
    };

    var btn = document.getElementById('gee-composite-btn');
    btn.disabled = true;
    btn.textContent = '合成中...';

    try {
        var resp = await fetch('/api/gee/composite', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            var err = await resp.json();
            throw new Error(err.detail || '合成失败');
        }

        var data = await resp.json();
        alert(data.message);
        renderGeeTasks(data.tasks);
        document.getElementById('gee-tasks').style.display = 'block';
    } catch (e) {
        alert('合成失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '📦 按周期合成下载';
    }
}

// ── 在地图上显示影像 ────────────────────────────────────

async function geeShowOnMap(idx) {
    var img = geeSearchResults[idx];
    if (!img) return;

    var collection = document.getElementById('gee-collection').value;
    var coordEl = document.getElementById('gee-coord');
    coordEl.textContent = '⏳ 加载影像...';

    try {
        var resp = await fetch('/api/gee/tile', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ collection: collection, image_id: img.id }),
        });

        if (!resp.ok) {
            var err = await resp.json();
            throw new Error(err.detail || '加载失败');
        }

        var data = await resp.json();

        // Remove previous tile layer
        if (geeTileLayer) {
            geeMap.removeLayer(geeTileLayer);
        }

        geeTileLayer = L.tileLayer(data.tile_url, {
            opacity: 0.8,
            maxZoom: 18,
        }).addTo(geeMap);

        coordEl.textContent = '🗺️ 已加载: ' + img.id;
    } catch (e) {
        coordEl.textContent = '❌ 加载失败: ' + e.message;
    }
}

// ── 批量下载 ────────────────────────────────────────────

async function geeDownload() {
    var ids = getSelectedImageIds();
    if (ids.length === 0) {
        alert('请先选择要下载的影像');
        return;
    }

    var lon = parseFloat(document.getElementById('gee-lon').value);
    var lat = parseFloat(document.getElementById('gee-lat').value);
    if (isNaN(lon) || isNaN(lat)) {
        alert('请先选择位置');
        return;
    }

    var body = {
        collection: document.getElementById('gee-collection').value,
        image_ids: ids,
        lon: lon,
        lat: lat,
        buffer_km: parseFloat(document.getElementById('gee-buffer').value) || 20,
        bands: document.getElementById('gee-bands').value.trim(),
        scale: document.getElementById('gee-scale').value || '10',
        cloud_mask: document.getElementById('gee-cloud-mask').checked,
        add_ndvi: document.getElementById('gee-add-ndvi').checked,
    };

    var btn = document.querySelector('.gee-btn.success');
    btn.disabled = true;
    btn.textContent = '提交中...';

    try {
        var resp = await fetch('/api/gee/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });

        if (!resp.ok) {
            var err = await resp.json();
            throw new Error(err.detail || '下载失败');
        }

        var data = await resp.json();
        alert(data.message);

        renderGeeTasks(data.tasks);
        document.getElementById('gee-tasks').style.display = 'block';

    } catch (e) {
        alert('下载失败: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.textContent = '📥 批量下载';
    }
}

// ── 任务状态 ────────────────────────────────────────────

function renderGeeTasks(tasks) {
    var list = document.getElementById('gee-tasks-list');
    list.innerHTML = '';

    // Link to GEE tasks page
    var linkRow = document.createElement('div');
    linkRow.className = 'gee-task-item';
    linkRow.style.justifyContent = 'center';
    linkRow.innerHTML =
        '<a href="https://code.earthengine.google.com/tasks" target="_blank" ' +
        'style="color:#0071e3;font-weight:600;text-decoration:none;">' +
        '📋 在 GEE 平台查看导出进度 →</a>';
    list.appendChild(linkRow);

    tasks.forEach(function(t) {
        if (t.error) return;
        var item = document.createElement('div');
        item.className = 'gee-task-item';
        item.innerHTML =
            '<span>' + escapeHtml(t.image_id || '') + '</span>' +
            '<span class="gee-task-state ' + (t.status || '') + '">' + (t.status || 'UNKNOWN') + '</span>';
        list.appendChild(item);
    });
}

async function geeRefreshTasks() {
    try {
        var resp = await fetch('/api/gee/tasks');
        var data = await resp.json();
        var list = document.getElementById('gee-tasks-list');
        list.innerHTML = '';

        // Link to GEE tasks page
        var linkRow = document.createElement('div');
        linkRow.className = 'gee-task-item';
        linkRow.style.justifyContent = 'center';
        linkRow.innerHTML =
            '<a href="https://code.earthengine.google.com/tasks" target="_blank" ' +
            'style="color:#0071e3;font-weight:600;text-decoration:none;">' +
            '📋 在 GEE 平台查看导出进度 →</a>';
        list.appendChild(linkRow);

        (data.tasks || []).forEach(function(t) {
            if (t.error) return;
            var item = document.createElement('div');
            item.className = 'gee-task-item';
            item.innerHTML =
                '<span>' + escapeHtml(t.description || '') + '</span>' +
                '<span class="gee-task-state ' + (t.state || '') + '">' + (t.state || 'UNKNOWN') + '</span>';
            list.appendChild(item);
        });

        document.getElementById('gee-tasks').style.display = 'block';
    } catch (e) {
        alert('查询任务状态失败: ' + e.message);
    }
}

// ── SHP 上传 ────────────────────────────────────────────

async function geeUploadShp(input) {
    var files = input.files;
    if (!files || files.length === 0) return;

    var formData = new FormData();
    for (var i = 0; i < files.length; i++) {
        formData.append('files', files[i]);
    }

    var btn = document.getElementById('gee-shp-btn');
    btn.textContent = '上传中...';

    try {
        var resp = await fetch('/api/gee/upload-shp', {
            method: 'POST',
            body: formData,
        });

        if (!resp.ok) {
            var err = await resp.json();
            throw new Error(err.detail || '上传失败');
        }

        var data = await resp.json();

        if (geeShpLayer) {
            geeMap.removeLayer(geeShpLayer);
        }

        geeShpLayer = L.geoJSON(data.geojson, {
            style: {
                color: '#0071e3',
                weight: 2,
                opacity: 0.8,
                fillOpacity: 0.15,
            },
        }).addTo(geeMap);

        if (data.bounds) {
            var b = data.bounds;
            geeMap.fitBounds([[b[1], b[0]], [b[3], b[2]]]);
        }

        document.getElementById('gee-coord').textContent =
            '📤 SHP: ' + data.file + ' (' + data.feature_count + ' 个要素)';

    } catch (e) {
        alert('SHP 上传失败: ' + e.message);
    } finally {
        btn.textContent = '📤 上传 SHP';
        input.value = '';
    }
}

// ── GEE 认证 ────────────────────────────────────────────

async function geeCheckAuth() {
    var badge = document.getElementById('gee-auth-badge');
    try {
        var resp = await fetch('/api/gee/auth-status');
        var data = await resp.json();
        if (data.authenticated) {
            badge.textContent = '✓ 已认证';
            badge.className = 'gee-auth-badge ok';
        } else {
            badge.textContent = '✗ 未认证';
            badge.className = 'gee-auth-badge error';
        }
    } catch (e) {
        badge.textContent = '检查失败';
        badge.className = 'gee-auth-badge error';
    }
}

async function geeAuthenticate() {
    var projectId = document.getElementById('gee-project-id').value.trim();
    var resultDiv = document.getElementById('gee-auth-result');
    resultDiv.style.display = 'block';
    resultDiv.textContent = '认证中...';

    try {
        var resp = await fetch('/api/gee/auth', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ project_id: projectId }),
        });

        var data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || '认证失败');
        }

        resultDiv.textContent = data.message || '认证成功';
        resultDiv.style.background = '#d4edda';
        resultDiv.style.color = '#155724';

        // Update badge
        geeCheckAuth();
    } catch (e) {
        resultDiv.textContent = '认证失败: ' + e.message;
        resultDiv.style.background = '#f8d7da';
        resultDiv.style.color = '#721c24';
    }
}

// ── 输出影像 ────────────────────────────────────────────

async function loadOutputImages() {
    var grid = document.getElementById('output-images');
    grid.innerHTML = '<p style="color:#888">加载中...</p>';

    try {
        var resp = await fetch('/api/tools');
        grid.innerHTML = '<p style="color:#888">分析结果会保存在 output/ 目录。在对话分析中进行遥感分析后，生成的影像会显示在这里。</p>';
    } catch (e) {
        grid.innerHTML = '<p style="color:#c00">加载失败</p>';
    }
}

// ── 工具函数 ────────────────────────────────────────────

function escapeHtml(text) {
    var div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
