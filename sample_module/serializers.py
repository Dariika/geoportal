from rest_framework import serializers
from common import serializers as geo_serializers

class DrawZoneSerializer(serializers.Serializer):
    h = serializers.IntegerField()
    point_start = geo_serializers.PointField()



    