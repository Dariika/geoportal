from django.shortcuts import render
from common.commands import CommandResponse, CommandView
from .serializers import DrawZoneSerializer
from common.models import VectorFeaturePO
from django.contrib.gis.geos import Polygon, Point
from .NewModel import main_model

class DrawZone(CommandView):
    serializer_class = DrawZoneSerializer
    name="Рисует зону поражения"
    description = "Рисует зону поражения"
    alias = "Определить зону"

    def handler(self, validated_data):
        result = main_model(validated_data)
        circle = Polygon(result)
        return CommandResponse([], [
            VectorFeaturePO({}, 
                            (circle))
        ], [])



