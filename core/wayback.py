"""Roof map imagery date selector (Esri World Imagery Wayback): one browser script, DISPLAY ONLY.

Two fixed imagery options (WAYBACK_OPTIONS): the live layer and one Wayback release. Python never fetches a tile;
the choice only switches the tile layer's URL in the user's browser. Nothing is uploaded, stored on disk or sent to
any model. The choice is kept per building in plain JS memory on window.parent (gone when the page is closed) and is
applied right after the tile layer is created, before the first tile load of a rebuilt map.

The raw tiles are shown as delivered: no image processing of any kind.
Docs: https://leafletjs.com/reference.html#tilelayer-seturl  https://leafletjs.com/reference.html#control-attribution
https://leafletjs.com/reference.html#domevent-disableclickpropagation
"""
from branca.element import MacroElement
from jinja2 import Template

LIVE = "live"
# (label, release id, URL template); the live entry uses the tile layer's own URL (core.config.ESRI_TILES_URL)
WAYBACK_OPTIONS = (
    ("Güncel (2025-11-20)", LIVE, None),
    (
        "2023-12-07", "56102",
        "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/WMTS/1.0.0/default028mm/MapServer/tile/56102/{z}/{y}/{x}",
    ),
)
WAYBACK_ATTRIBUTION = "Esri World Imagery Wayback ({date}) — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
CAPTION = (
    "Görüntü tarihi: haritanın sağ üstünden farklı yıllara ait çekimleri seçebilirsiniz; en net olanı seçin. "
    "Yüksek binalarda çerçeveyi yeniden hizalamanız gerekebilir."
)


class WaybackScript(MacroElement):
    """Imagery date selector for one building. A stored choice is applied with setUrl right after the tile layer is
    created (same script tick, before any paint and before the fitted view's tiles are requested)."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            var map = {{ this._parent.get_name() }}, tiles = {{ this.tile_layer_name }};
            var KEY = {{ this.key|tojson }}, LIVE = {{ this.live|tojson }}, LIVE_URL = {{ this.live_url|tojson }};
            var OPTIONS = {{ this.options|tojson }};  // [{label, release, url}], the live option first
            var WB_ATTR = {{ this.wb_attr|tojson }};
            var LIVE_ATTR = tiles.getAttribution(), currentUrl = LIVE_URL, currentAttr = LIVE_ATTR;

            function store() {  // a Map on window.parent; null when it is unreachable
                try {
                    var w = window.parent;
                    if (!w.__roofWayback) w.__roofWayback = new w.Map();
                    return w.__roofWayback;
                } catch (err) { return null; }
            }
            function load() {  // the stored release id of this building, or null
                try { var s = store(); return s && s.has(KEY) ? String(s.get(KEY)) : null; } catch (err) { return null; }
            }
            function save(release) {
                try { var s = store(); if (s) s.set(KEY, release); } catch (err) {}
            }
            function find(release) { return OPTIONS.filter(function (o) { return o.release === release; })[0] || null; }
            function apply(choice) {  // an option of OPTIONS; the live one (or null) is the original layer
                var wb = !!(choice && choice.release !== LIVE && choice.url);
                var url = wb ? choice.url : LIVE_URL, attr = wb ? WB_ATTR.replace("{date}", choice.label) : LIVE_ATTR;
                if (url !== currentUrl) {
                    tiles.setUrl(url);
                    currentUrl = url;
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

            var chosen = find(load()) || OPTIONS[0];
            if (chosen.release !== LIVE) { try { apply(chosen); } catch (err) {} }  // before the first tile load

            var DateControl = L.Control.extend({
                options: {position: "topright"},
                onAdd: function () {
                    var div = L.DomUtil.create("div", "leaflet-bar roof-wayback");
                    div.style.cssText = "background:#fff;padding:4px 6px;font:12px/1.4 sans-serif;color:#111;";
                    var label = L.DomUtil.create("div", "", div);
                    label.textContent = "Görüntü tarihi";
                    label.style.fontWeight = "600";
                    var sel = L.DomUtil.create("select", "", div);
                    OPTIONS.forEach(function (o) {
                        var opt = document.createElement("option");
                        opt.value = o.release; opt.textContent = o.label;
                        sel.appendChild(opt);
                    });
                    sel.value = chosen.release;
                    L.DomEvent.on(sel, "change", function () {
                        var o = find(sel.value);
                        try { apply(o); save(o.release); } catch (err) { console.warn("roofWayback:", err); }
                    });
                    L.DomEvent.disableClickPropagation(div);  // never reaches the map, Draw or click-to-align
                    L.DomEvent.disableScrollPropagation(div);
                    return div;
                }
            });
            map.whenReady(function () {
                try { new DateControl().addTo(map); } catch (err) { console.warn("roofWayback:", err); }
            });
        })();
        {% endmacro %}
        """
    )

    def __init__(self, tile_layer, key, live_url):
        super().__init__()
        self._name = "WaybackScript"
        self.tile_layer_name, self.key, self.live, self.live_url = tile_layer.get_name(), str(key), LIVE, live_url
        self.options = [{"label": label, "release": release, "url": url} for label, release, url in WAYBACK_OPTIONS]
        self.wb_attr = WAYBACK_ATTRIBUTION


def add_to_map(m, tile_layer, key, live_url):
    """Add the date selector to the folium map `m` right after `tile_layer` (the Esri folium.TileLayer) is added.
    `key`: stable building key; `live_url`: the tile layer's URL template (the live option)."""
    m.add_child(WaybackScript(tile_layer, key, live_url))
