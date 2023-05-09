from django import template
from django.template import loader
from rest_framework.renderers import HTMLFormRenderer
from common import serializers
register = template.Library()

@register.simple_tag
def render_geofield(field, style):
    if(issubclass(type(field._field), serializers.BaseGeometryField)):
        geo_template = loader.get_template("geoportal_core/geofield.html")
        return geo_template.render({'field': field.as_form_field(), 'geom_type': field._field.geom_type})
    renderer = style.get('renderer', HTMLFormRenderer())
    return renderer.render_field(field, style)

@register.simple_tag
def render_geoform(serializer):
    geo_template = loader.get_template("geoportal_core/geoform.html")
    return geo_template.render({'form': serializer.data.serializer, 
                                'style': {'template_pack': 'rest_framework/vertical'}})