from rest_framework import serializers
from rest_framework_gis.fields import GeometryField

class MyGISModel(serializers.Serializer):
    int_property = serializers.IntegerField()
    string_property = serializers.CharField(max_length=20)
    point = GeometryField()


