from rest_framework_gis.fields import GeometryField
from django.core.exceptions import ValidationError

# rest_framework_gis рассчитан на сериализацию/десериализацию конкрентных моделей,
# и поэтому GeometryField никак не проверяет тип геометрии (в отличие от полей формы из contrib.gis)
# однако для команд модуля нужно определять сериализатор без привязки к django моделям
# поэтому пришлось определить проверку типа вручную (ориентируясь на contrib.gis.forms.fields)
class BaseGeometryField(GeometryField):
    geom_type = "Geometry"

    def to_internal_value(self, value):
        geom = super().to_internal_value(value)
        if geom is None:
            return geom
        
        geom_type = str(geom.geom_type).upper()
        if geom_type != self.geom_type.upper():
            raise ValidationError(
                f"invalid geometry type. Expected {self.geom_type}, got {geom_type}"
            )
        
        return geom
    

class GeometryCollectionField(BaseGeometryField):
    geom_type = "GeometryCollection"


class PointField(BaseGeometryField):
    geom_type = "Point"


class MultiPointField(BaseGeometryField):
    geom_type = "MultiPoint"


class LineStringField(BaseGeometryField):
    geom_type = "LineString"


class MultiLineStringField(BaseGeometryField):
    geom_type = "MultiLineString"


class PolygonField(BaseGeometryField):
    geom_type = "Polygon"


class MultiPolygonField(BaseGeometryField):
    geom_type = "MultiPolygon"