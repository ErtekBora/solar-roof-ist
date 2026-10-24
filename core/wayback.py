"""Roof map imagery date selector (Esri World Imagery Wayback): one browser script, DISPLAY ONLY.

Privacy/licence rule: Python never fetches a tile or the release list. The script runs in the user's browser: it
loads the Wayback release list, fetches the z19 tile under the building centre for the live layer and every release
(in memory only), hashes the bytes and keeps one release per distinct capture (the oldest one, the date the capture
first appeared, as the Wayback app's "local changes" list does). The user's choice switches the tile layer's URL;
nothing is uploaded, stored on disk or sent to any model. The choice and the computed list are kept per building in
plain JS memory on window.parent (gone when the page is closed). On any failure the control hides and the live layer
stays.

waybackconfig.json: {"<release>": {"itemTitle": "World Imagery (Wayback YYYY-MM-DD)",
"itemURL": ".../MapServer/tile/<release>/{level}/{row}/{col}", ...}, ...}
Docs: https://leafletjs.com/reference.html#tilelayer-seturl  https://leafletjs.com/reference.html#control-attribution
https://leafletjs.com/reference.html#domevent-disableclickpropagation  https://leafletjs.com/reference.html#map-project
https://developer.mozilla.org/en-US/docs/Web/API/SubtleCrypto/digest
https://developer.mozilla.org/en-US/docs/Web/API/AbortController
"""
from branca.element import MacroElement
from jinja2 import Template

CONFIG_URL = "https://s3-us-west-2.amazonaws.com/config.maptiles.arcgis.com/waybackconfig.json"
NATIVE_ZOOM = 19  # must match app.ESRI_NATIVE_ZOOM
MIN_TILE_BYTES = 3000  # smaller responses are Esri's "no data" placeholder
CONCURRENCY = 8  # tile requests in flight at once
TILE_TIMEOUT_MS = 8000  # per request
BUDGET_MS = 20000  # all tile requests; after it the list is built from what has arrived
WAYBACK_ATTRIBUTION = "Esri World Imagery Wayback ({date}) — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
CAPTION = (
    "Görüntü tarihi: haritanın sağ üstünden farklı yıllara ait çekimleri seçebilirsiniz; en net olanı seçin. "
    "Yüksek binalarda çerçeveyi yeniden hizalamanız gerekebilir."
)

# Pure functions (no DOM, no Leaflet, no Jinja syntax), so the same text can be run outside the map (e.g. with Node).
GROUPING_JS = """
function parseReleases(config) {  // waybackconfig.json -> [{release, date, url}] with a Leaflet {z}/{y}/{x} template
    var out = [];
    Object.keys(config || {}).forEach(function (id) {
        var item = config[id] || {}, m = /Wayback (\\d{4}-\\d{2}-\\d{2})/.exec(item.itemTitle || "");
        if (!m || typeof item.itemURL !== "string") return;
        var url = item.itemURL.replace("{release}", id).replace("{level}", "{z}").replace("{row}", "{y}").replace("{col}", "{x}");
        out.push({release: id, date: m[1], url: url});
    });
    return out;
}

function byDateDesc(a, b) { return a.date < b.date ? 1 : (a.date > b.date ? -1 : 0); }

// releases: parseReleases output; hashes: {release: hash} (missing or null = failed or no-data, skipped);
// liveHash: hash of the live tile or null. Returns the select options, "Güncel" first, then distinct captures by
// date descending; each capture is represented by its OLDEST release.
function groupReleases(releases, hashes, liveHash) {
    var groups = {};
    releases.forEach(function (r) {
        var h = hashes[r.release];
        if (!h) return;
        if (!groups[h] || r.date < groups[h].date) groups[h] = r;
    });
    var live = liveHash ? groups[liveHash] : null;
    var options = [{release: "live", date: live ? live.date : null, label: live ? "Güncel (" + live.date + ")" : "Güncel", url: null}];
    Object.keys(groups).filter(function (h) { return h !== liveHash; }).map(function (h) { return groups[h]; })
        .sort(byDateDesc).forEach(function (r) {
            options.push({release: r.release, date: r.date, label: r.date, url: r.url});
        });
    return options;
}
"""


class WaybackScript(MacroElement):
    """Imagery date selector for one building. A stored choice is applied with setUrl right after the tile layer is
    created (same script tick, before any paint and before the fitted view's tiles are requested); the option list is
    reused from window.parent or computed once in whenReady."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            {{ this.grouping }}
            var map = {{ this._parent.get_name() }}, tiles = {{ this.tile_layer_name }};
            var KEY = {{ this.key|tojson }}, CENTER = L.latLng({{ this.center|tojson }});
            var LIVE_URL = {{ this.live_url|tojson }}, CONFIG_URL = {{ this.config_url|tojson }};
            var NZ = {{ this.native }}, TILE = 256, MIN_BYTES = {{ this.min_bytes }};
            var CONCURRENCY = {{ this.concurrency }}, TIMEOUT_MS = {{ this.timeout_ms }}, BUDGET_MS = {{ this.budget_ms }};
            var WB_ATTR = {{ this.wb_attr|tojson }};
            var LIVE_ATTR = tiles.getAttribution(), currentUrl = LIVE_URL, currentAttr = LIVE_ATTR;
            map.roofImageryUrl = LIVE_URL;  // the current imagery template, read by core/superres

            function store(name) {  // a Map on window.parent; null when it is unreachable
                try {
                    var w = window.parent;
                    if (!w[name]) w[name] = new w.Map();
                    return w[name];
                } catch (err) { return null; }
            }
            function load(name) {  // values are JSON strings: no object from a removed iframe is kept
                try {
                    var s = store(name), v = s && s.get(KEY);
                    return v ? JSON.parse(v) : null;
                } catch (err) { return null; }
            }
            function save(name, value) {
                try { var s = store(name); if (s) s.set(KEY, JSON.stringify(value)); } catch (err) {}
            }
            function apply(choice) {  // an option of the list; "live" (or null) is the original layer
                var wb = !!(choice && choice.release !== "live" && choice.url);
                var url = wb ? choice.url : LIVE_URL, attr = wb ? WB_ATTR.replace("{date}", choice.date) : LIVE_ATTR;
                if (url !== currentUrl) {
                    tiles.setUrl(url);
                    currentUrl = map.roofImageryUrl = url;
                    map.fire("roofimagerychange", {url: url});
                }
                if (attr !== currentAttr) {
                    tiles.options.attribution = attr;
                    if (map.attributionControl) {
                        map.attributionControl.removeAttribution(currentAttr);
                        map.attributionControl.addAttribution(attr);
                    }
                    currentAttr = attr;
                }
            }

            var stored = load("__roofWayback");
            if (stored) { try { apply(stored); } catch (err) {} }  // before the first tile load of this build

            function fetchBytes(url, budget) {  // ArrayBuffer; aborted after TIMEOUT_MS or when the budget ends
                var ctrl = new AbortController(), timer = setTimeout(function () { ctrl.abort(); }, TIMEOUT_MS);
                if (budget) budget.addEventListener("abort", function () { ctrl.abort(); });
                return fetch(url, {signal: ctrl.signal, mode: "cors", credentials: "omit"}).then(function (r) {
                    if (!r.ok) throw new Error("HTTP " + r.status);
                    return r.arrayBuffer();
                }).finally(function () { clearTimeout(timer); });
            }
            function hex(buf) {
                return Array.prototype.map.call(new Uint8Array(buf), function (b) { return ("0" + b.toString(16)).slice(-2); }).join("");
            }
            function tileHash(url, budget) {  // null for a failed or no-data (< MIN_BYTES) response
                return fetchBytes(url, budget).then(function (buf) {
                    if (buf.byteLength < MIN_BYTES) return null;
                    return crypto.subtle.digest("SHA-1", buf).then(hex);
                }).catch(function () { return null; });
            }
            function runPool(jobs) {  // at most CONCURRENCY at once; resolves when all are done or after BUDGET_MS
                var budget = new AbortController();
                return new Promise(function (resolve) {
                    var results = new Array(jobs.length), next = 0, active = 0, done = false;
                    var timer = setTimeout(finish, BUDGET_MS);
                    function finish() {
                        if (done) return;
                        done = true; clearTimeout(timer); budget.abort();
                        resolve(results);
                    }
                    function pump() {
                        if (done) return;
                        if (next >= jobs.length && active === 0) { finish(); return; }
                        while (active < CONCURRENCY && next < jobs.length) {
                            (function (i) {
                                active++;
                                jobs[i](budget.signal).then(function (v) { results[i] = v; }, function () { results[i] = null; })
                                    .then(function () { active--; pump(); });
                            })(next++);
                        }
                    }
                    pump();
                });
            }
            function computeOptions() {
                var p = map.project(CENTER, NZ), xy = {z: NZ, x: Math.floor(p.x / TILE), y: Math.floor(p.y / TILE)};
                return fetchBytes(CONFIG_URL, null).then(function (buf) {
                    var releases = parseReleases(JSON.parse(new TextDecoder().decode(buf)));
                    var jobs = [function (b) { return tileHash(L.Util.template(LIVE_URL, xy), b); }].concat(
                        releases.map(function (r) { return function (b) { return tileHash(L.Util.template(r.url, xy), b); }; }));
                    return runPool(jobs).then(function (res) {
                        var hashes = {};
                        releases.forEach(function (r, i) { hashes[r.release] = res[i + 1] || null; });
                        return groupReleases(releases, hashes, res[0] || null);
                    });
                });
            }

            var DateControl = L.Control.extend({
                options: {position: "topright"},
                onAdd: function () {
                    var div = L.DomUtil.create("div", "leaflet-bar roof-wayback");
                    div.style.cssText = "background:#fff;padding:4px 6px;font:12px/1.4 sans-serif;color:#111;";
                    var label = L.DomUtil.create("div", "", div);
                    label.textContent = "Görüntü tarihi";
                    label.style.fontWeight = "600";
                    this.status = L.DomUtil.create("div", "", div);
                    this.status.textContent = "Görüntü tarihleri yükleniyor…";
                    this.select = L.DomUtil.create("select", "", div);
                    this.select.style.display = "none";
                    L.DomEvent.disableClickPropagation(div);  // never reaches the map, Draw or click-to-align
                    L.DomEvent.disableScrollPropagation(div);
                    return div;
                }
            });
            var control = null;
            function hide() { if (control) { map.removeControl(control); control = null; } }
            function fill(options) {
                if (!control || options.length < 2) { hide(); return; }
                var sel = control.select, chosen = stored ? stored.release : "live";
                if (!options.some(function (o) { return o.release === chosen; })) chosen = "live";
                options.forEach(function (o) {
                    var opt = document.createElement("option");
                    opt.value = o.release; opt.textContent = o.label;
                    sel.appendChild(opt);
                });
                sel.value = chosen;
                L.DomEvent.on(sel, "change", function () {
                    var o = options.filter(function (x) { return x.release === sel.value; })[0];
                    try { apply(o); save("__roofWayback", o); } catch (err) { console.warn("roofWayback:", err); }
                });
                control.status.style.display = "none";
                sel.style.display = "";
            }

            map.whenReady(function () {
                try {
                    control = new DateControl();
                    control.addTo(map);
                    var list = load("__roofWaybackList");
                    if (list) { fill(list); return; }  // computed before for this building: no refetch
                    if (!(window.crypto && crypto.subtle && window.fetch && window.AbortController)) { hide(); return; }
                    computeOptions().then(function (options) {
                        if (options.length >= 2) save("__roofWaybackList", options);
                        fill(options);
                    }).catch(function (err) { console.warn("roofWayback: date list unavailable:", err); hide(); });
                } catch (err) { console.warn("roofWayback:", err); hide(); }
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, tile_layer, key, center, live_url):
        super().__init__()
        self._name = "WaybackScript"
        self.grouping, self.tile_layer_name = GROUPING_JS, tile_layer.get_name()
        self.key, self.center, self.live_url = str(key), [float(center[0]), float(center[1])], live_url
        self.config_url, self.native, self.min_bytes = CONFIG_URL, NATIVE_ZOOM, MIN_TILE_BYTES
        self.concurrency, self.timeout_ms, self.budget_ms = CONCURRENCY, TILE_TIMEOUT_MS, BUDGET_MS
        self.wb_attr = WAYBACK_ATTRIBUTION


def add_to_map(m, tile_layer, key, center, live_url):
    """Add the date selector to the folium map `m` right after `tile_layer` (the Esri folium.TileLayer) is added.
    `key`: stable building key; `center`: (lat, lng) of the building; `live_url`: the tile layer's URL template."""
    m.add_child(WaybackScript(tile_layer, key, center, live_url))
