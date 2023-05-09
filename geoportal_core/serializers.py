from rest_framework import serializers
from .models import GISModule, Area, Layer, VectorFeature
from rest_framework_gis.serializers import GeoFeatureModelSerializer

class ModuleListSerializer(serializers.ModelSerializer):
    owner = serializers.PrimaryKeyRelatedField(read_only=True)

    class Meta:
        model = GISModule
        fields = '__all__'

class AreaListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Area
        fields = ['name', 'alias', 'bbox']

class LayerSerializer(serializers.ModelSerializer):
    class Meta:
        model = Layer
        fields = ['name', 'alias', 'ordering', 'layer_type']

class VectorFeatureSerializer(GeoFeatureModelSerializer):
    class Meta:
        model = VectorFeature
        fields = ['properties', 'geometry', 'datetime']
        geo_field = 'geometry'
        auto_bbox = True

    def get_properties(self, instance, fields):
        p = instance.properties
        p["datetime"] = instance.datetime
        return p