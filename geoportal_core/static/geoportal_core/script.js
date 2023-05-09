ol.proj.useGeographic();

var vectorGroup = new ol.layer.Group({
    title: 'Векторные слои',
    layers: new ol.Collection()
});

var areaLayer = new ol.layer.Vector({
    title: 'Область',
    style: new ol.style.Style({
        stroke: new ol.style.Stroke({
          color: 'orange',
          width: 7,
          lineDash: [10, 12]
        }),
      }),
    source: new ol.source.Vector()
})

var drawingInteraction = null

var mapView = new ol.View({
    center: [34, 68],
    zoom: 7,
});

var map = new ol.Map({
    target: 'map',
    layers: [
        new ol.layer.Group({
            title: 'Подложка',
            type: 'base',
            layers: [
                new ol.layer.Tile({
                    title: 'OSM',
                    source: new ol.source.OSM()
                })
            ]
        }), 
        vectorGroup,
        areaLayer,
    ],
    view: mapView
  });

var layerSwitcher = new ol.control.LayerSwitcher({
    startActive: true,
    activationMode: 'click'
});

map.addControl(layerSwitcher);

var HOST = "http://"+window.location.host
var MODULE = window.location.pathname.split('/')[2]

uploadArea(MODULE, $("#areas").val())

$("#areas").change(function(){
   uploadArea(MODULE, $(this).val())
})

function uploadArea(module_name, name){
    $.ajax(`${HOST}/modules/${module_name}/areas/${name}/`)
    .done(function(data){
        zoomToArea(data);
        uploadVectorLayers(module_name, name)
    })
    .fail(function (t) {
        console.log(`error loading area ${name}`)
    });
}

function zoomToArea(area){
    areaLayer.getSource().clear();
    var poly = new ol.geom.Polygon(area.bbox.coordinates)
    areaLayer.getSource().addFeature(new ol.Feature({
        geometry: poly,
        name: area.alias
    }));
    mapView.fit(poly, {duration: 800})
}

function uploadVectorLayers(module_name, area_name){
    vectorGroup.getLayers().clear()
    $.getJSON(`${HOST}/modules/${module_name}/areas/${area_name}/layers/`, function (data){
        for(var layer of data){
            if (layer.layer_type == 'V'){
                var mapLayer = new ol.layer.Vector({
                    title: layer.alias,
                    source: new ol.source.Vector({
                        format: new ol.format.GeoJSON(),
                        url: `${HOST}/modules/${module_name}/areas/${area_name}/layers/${layer.name}/`
                    })
                });
                vectorGroup.getLayers().insertAt(0, mapLayer);
            } 
        } 
        layerSwitcher.renderPanel();
    });
}

$(".command_form").submit(function(event){
    event.preventDefault();
    var command_name = $(this).attr("name")
    $.ajax({
        url: `${HOST}/modules/${MODULE}/commands/${command_name}/`,
        data: $(this).serialize()
    })
    .done(function (data){
        map.addLayer(new ol.layer.Vector({
            title: `Команда ${command_name}`,
            source: new ol.source.Vector({
                features: new ol.format.GeoJSON().readFeatures(data.gis_data)
            })
        }));
        layerSwitcher.renderPanel();
    })
    .fail(function (data){
        console.log("failed");
        console.log(data);
    })
});

function clearGeoField(layer, text_area, status_span){
    status_span.text("Не выбрано");
    if(status_span.hasClass("status-selected")){
        status_span.removeClass("status-selected");
    }
    text_area.text("");
    layer.getSource().clear();
}

//var drawingLayers = []

$(".command_form").children(".geofield").each(function(){
    // TODO: add layers only when form is show, then remove
    var drawingSource = new ol.source.Vector();
    var drawingLayer = new ol.layer.Vector({
        source: drawingSource
    });
    map.addLayer(drawingLayer);
    var text_area = $(this).children("textarea").first()
    var geocontrol = $(this).children(".geo-edit").first()
    var geom_type = geocontrol.attr('geom_type')
    var status = geocontrol.children("span").first();
    geocontrol.children(".add").click(function(){
        clearGeoField(drawingLayer, text_area, status);
        if (drawingInteraction != null){
            return;
        }
        drawingInteraction = new ol.interaction.Draw({
            source: drawingSource,
            type: geom_type,
            maxPoints: 6
        });
        drawingInteraction.on('drawend', function(event){
            text_area.text(new ol.format.WKT().writeFeature(event.feature))
            status.text("Выбрано")
            status.toggleClass("status-selected");
            map.removeInteraction(drawingInteraction);
            drawingInteraction = null
        })
        map.addInteraction(drawingInteraction);
    })
    geocontrol.children(".clear").click(function(){
        clearGeoField(drawingLayer, text_area, status);
    })
});


$("#commands").val(-1);

$("#commands").change(function(){
    var val = $(this).val();
    $(".command_form_wrapper").hide();
    if(val != -1){
        $(`.command_form_wrapper[name='${val}']`).show();
    }
})