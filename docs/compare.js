
const MAIN_LAYER_INDEX = 'Main Index Grid';
const JICA_LAYER_INDEX = 'JICA Index Grid';
const BORDER_LAYER_INDEX = 'Border Index Grid';

const boundaryStyle = new ol.style.Style({
    stroke: new ol.style.Stroke({
        color: 'black',
        width: 2,
    }),
    fill: new ol.style.Fill({
        color: 'rgba(255,255,255,0.0)',
    }),
});

const gridStyle = new ol.style.Style({
    stroke: new ol.style.Stroke({
        color: 'black',
        width: 1,
    }),
    fill: new ol.style.Fill({
        color: 'rgba(255,255,255,0.0)',
    }),
});

const NEPAL_CENTER = [85.00, 28.00];

function getMap(target, layers) {
    return new ol.Map({
        controls: [],
        interactions: getInteractions(),
        target: target,
        view: new ol.View({
            zoom: 7,
            maxZoom: 15,
            center: NEPAL_CENTER
        }),
    });
}

const topo_attribution = makeLink("https://www.maze.com.np/Maps/Topo-Nepal", "The Maze (corrected) ");

function getTopoLayer(type) {
    const src = new ol.source.XYZ({
        url: `https://indianopenmaps.fly.dev/nepal/topo/maze/${type}/{z}/{x}/{y}.webp`,
        attributions: [topo_attribution],
    });
    return new ol.layer.Tile({
        background: 'grey',
        source: src,
        maxZoom: 15,
    });
}

const border_attribution = makeLink("https://pahar.in/", "Pahar");

function getBorderLayer() {
    const src = new ol.source.XYZ({
        url: `https://indianopenmaps.fly.dev/nepal/topo/border/{z}/{x}/{y}.webp`,
        attributions: [border_attribution],
    });
    return new ol.layer.Tile({
        background: 'grey',
        source: src,
        maxZoom: 14,
    });
}

function getGridSourceMain() {
    const src = new ol.source.Vector({
        format: new ol.format.GeoJSON(),
        url: 'index_main.geojson',
        overlaps: false,
        attributions: [makeLink("https://www.maze.com.np/Maps/Topo-Nepal", "The Maze (corrected)")]
    });
    return src;
}

function getGridSourceJICA() {
    const src = new ol.source.Vector({
        format: new ol.format.GeoJSON(),
        url: 'index_jica.geojson',
        overlaps: false,
        attributions: []
    });
    return src;
}

function getGridSourceBorder() {
    const src = new ol.source.Vector({
        format: new ol.format.GeoJSON(),
        url: 'index_border.geojson',
        overlaps: false,
        attributions: []
    });
    return src;
}


function getLayer(src, style, visible) {
    const layer = new ol.layer.Vector({
        visible: visible,
        source: src,
        style: gridStyle
    });
    return layer;
}


function getLayerGroup() {

    const osmLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png', 
            attributions: [
                '&copy; ' + makeLink('https://www.openstreetmap.org/copyright', 'OpenStreetMap contributors') 
            ],
        }),
        baseLayer: true,
        visible: true,
        maxZoom: 19,
        title: 'OpenStreetMap',
    });
    const gStreetsLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://mt{0-3}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
            attributions: [
                'Map data &copy; 2023 Google'
            ]
        }),
        baseLayer: true,
        visible: false,
        maxZoom: 20,
        title: 'Google Streets',
    });
    const gHybridLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://mt{0-3}.google.com/vt/lyrs=s,h&x={x}&y={y}&z={z}',
            attributions: [
                'Map data &copy; 2023 Google',
                'Imagery &copy; 2023 TerraMetrics'
            ]
        }),
        baseLayer: true,
        visible: false,
        maxZoom: 20,
        title: 'Google Hybrid',
    });
    const esriWorldLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attributions: [
                'Tiles &copy; Esri',
                'Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, ' +
                'Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
            ]
        }),
        baseLayer: true,
        visible: false,
        maxZoom: 20,
        title: 'ESRI Satellite',
    });
    const otmLayer = new ol.layer.Tile({
        source: new ol.source.XYZ({
            url: 'https://{a-c}.tile.opentopomap.org/{z}/{x}/{y}.png', 
            attributions: [
                'Map data: &copy; ' + makeLink('https://www.openstreetmap.org/copyright', 'OpenStreetMap contributors'),
                makeLink('http://viewfinderpanoramas.org', 'SRTM'),
                'Map style: &copy; ' +
                makeLink('https://opentopomap.org', 'OpenTopoMap') +
                ' (' + makeLink('https://creativecommons.org/licenses/by-sa/3.0/', 'CC-BY-SA') + ')'
            ]
        }),
        baseLayer: true,
        visible: false,
        maxZoom: 17,
        title: 'OpenTopoMap'
    });

    return new ol.layer.Group({
        title: 'Base Layers',
        openInLayerSwitcher: true,
        layers: [
            gHybridLayer,
            gStreetsLayer,
            esriWorldLayer,
            otmLayer,
            osmLayer,
        ]
    });
}

document.addEventListener("DOMContentLoaded", () => {

    var statusElem = document.getElementById('call_status');
    var setStatus = (msg, err) => {
        var alreadyError = false;
        const prevMsg = statusElem.innerHTML;
        if (statusElem.hasAttribute("class")) {
            alreadyError = true;
        }
        if (err === true) {
            if (alreadyError === true) {
                msg = prevMsg + '<br>' + msg;
            } else {
                statusElem.setAttribute("class", "error");
            }
            statusElem.innerHTML = msg;
        } else if (alreadyError !== true) {
            statusElem.removeAttribute("class");
            statusElem.innerHTML = msg;
        }
    };

    ol.proj.useGeographic();

    var map1 = getMap('map1');
    var map2 = getMap('map2');

    const mainTopoLayer = getTopoLayer('main');
    mainTopoLayer.set('title', 'Main Survey');
    mainTopoLayer.set('visible', true);

    const jicaTopoLayer = getTopoLayer('jica');
    jicaTopoLayer.set('title', 'JICA Survey');
    jicaTopoLayer.set('visible', false);

    const borderLayer = getBorderLayer();
    borderLayer.set('title', 'Border Survey');
    borderLayer.set('visible', false);

    const map1LayerGroup = new ol.layer.Group({
        title: 'Left Map Source',
        openInLayerSwitcher: true,
        layers: [borderLayer, jicaTopoLayer, mainTopoLayer]
    });
    map1.addLayer(map1LayerGroup);

    const map1ControlLayerGroup = new ol.layer.Group({
        title: 'Left Map Source',
        openInLayerSwitcher: true,
        layers: [
            new ol.layer.Tile({
                title: 'Border Survey',
                visible: false,
            }),
            new ol.layer.Tile({
                title: 'JICA Survey',
                visible: false,
            }),
            new ol.layer.Tile({
                title: 'Main Survey',
                visible: true,
            })
        ]
    });

    map1ControlLayerGroup.getLayers().forEach(l => {
        l.on('change:visible', (e) => {
            const title = e.target.get('title');
            const visible = e.target.getVisible();
            map1LayerGroup.getLayers().forEach(map1_l => {
                if (map1_l.get('title') === title) {
                    map1_l.setVisible(visible);
                }
            });
        });
    });
    map2.addLayer(map1ControlLayerGroup);

    map2.addLayer(new ol.layer.Vector({
        source: new ol.source.Vector({
            attributions: [topo_attribution]
        })}
    ));
    const layerGroup = getLayerGroup();
    map2.addLayer(layerGroup);

    var compareElem = document.getElementById('compare');

    const mainGridSrc = getGridSourceMain();
    const jicaGridSrc = getGridSourceJICA();
    const borderGridSrc = getGridSourceBorder();

    var getGridLayer = (layer_index, gridSrc) => {
        return new ol.layer.Vector({
            title: layer_index,
            visible: false,
            source: gridSrc,
            style: gridStyle,
            displayInLayerSwitcher: true
        });
    };
    const gridMainLayer1 = getGridLayer(MAIN_LAYER_INDEX, mainGridSrc);
    const gridMainLayer2 = getGridLayer(MAIN_LAYER_INDEX, mainGridSrc);
    map1.addLayer(gridMainLayer1);
    map2.addLayer(gridMainLayer2);

    const gridJICALayer1 = getGridLayer(JICA_LAYER_INDEX, jicaGridSrc);
    const gridJICALayer2 = getGridLayer(JICA_LAYER_INDEX, jicaGridSrc);
    map1.addLayer(gridJICALayer1);
    map2.addLayer(gridJICALayer2);

    const gridBorderLayer1 = getGridLayer(BORDER_LAYER_INDEX, borderGridSrc);
    const gridBorderLayer2 = getGridLayer(BORDER_LAYER_INDEX, borderGridSrc);
    map1.addLayer(gridBorderLayer1);
    map2.addLayer(gridBorderLayer2);

    function showPopup(map, e, pop, key, contentFn) {
        var features = map.getFeaturesAtPixel(e.pixel);
        features = features.filter((f) => f.get(key));
        const feature = features.length ? features[0] : undefined;
        if (feature === undefined) {
            pop.hide();
            return;
        }
        //console.log(feature.getGeometry().getExtent());
        const html = contentFn(feature);
        if (html === null) {
            pop.hide();
            return;
        }
        pop.show(e.coordinate, html);
    }

    function addPopup(layer, map, type) {
      var popup = new ol.Overlay.Popup({
        popupClass: "default", //"tooltips", "warning" "black" "default", "tips", "shadow",
        closeBox: true,
        positioning: 'center-left',
        autoPan: {
          animation: { duration: 250 }
        }
      });
      map.addOverlay(popup);
      map.on('click', function(e) {
        if (!layer.getVisible()) {
            return;
        }
        const key = type == 'border' ? 'id' : 'Name';
        showPopup(map, e, popup, key, (f) => {
            var sheetNo = f.get(key);
         
            return '<b text-align="center">' + sheetNo + '</b>';
        });
      });

      return popup;
    }

    addPopup(gridJICALayer2, map2, 'jica');
    addPopup(gridMainLayer2, map2, 'main');
    addPopup(gridBorderLayer2, map2, 'border');

    var swipe = new ol.control.SwipeMap({ right: true });

    function tickleSwipe() {
      const pos = swipe.get('position');
      console.log('tickle', pos);
      swipe.set('position', pos - 0.000001);
    }
    map2.on('change:size', function(e) {
        // hack to trigger redraw on fullscreen and inital render
        console.log('change event fired:', e);
        tickleSwipe();
    });


    map1.addInteraction(new ol.interaction.Synchronize({ maps: [map2] }));
    map2.addInteraction(new ol.interaction.Synchronize({ maps: [map1] }));
    var layerSwitcher = new ol.control.LayerSwitcher({
        reordering: false,
        noScroll: true,
        mouseover: true,
    });
    layerSwitcher.on('layer:visible', (e) => {
        // console.log('layer:visible', e);
        const l = e.layer;
        if (l.get('title') === MAIN_LAYER_INDEX) {
            gridMainLayer1.setVisible(l.getVisible());
        }
        if (l.get('title') === JICA_LAYER_INDEX) {
            gridJICALayer1.setVisible(l.getVisible());
        }
        if (l.get('title') === BORDER_LAYER_INDEX) {
            gridBorderLayer1.setVisible(l.getVisible());
        }
    });
    // layerSwitcher.on('select', (e) => {
    //    console.log(e);
    // });
    map2.addControl(layerSwitcher);

    map2.addControl(new ol.control.FullScreen({ source: 'compare' }));
    map2.addControl(new ol.control.Zoom());
    map2.addControl(new ol.control.Attribution({ collapsed: true, collapsible: true }));
    // map1.addControl(new ol.control.Attribution({ collapsed: true, collapsible: true}));
    var currentMode;
    function setMode(mode) {
        if (mode) {
            currentMode = mode;
            // Remove tools
            map2.removeControl(swipe);
            // Set interactions
            switch (mode) {
                case 'swipev':
                case 'swipeh': {
                    map2.addControl(swipe);
                    swipe.set('orientation', (mode==='swipev' ? 'vertical' : 'horizontal'));
                    break;
                }
            }
            // Update position
            document.getElementById("compare").className = mode;
        }
        map1.updateSize();
        map2.updateSize();
    }
    setMode('swipev');
});

