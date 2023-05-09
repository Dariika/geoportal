from datetime import datetime

class AreaPO:
    def __init__(self, name: str, bbox: tuple, alias:str = ""):
        """Именнованая прямоугольная область на карте, 
        к которой могут быть привязаны слои и пространственные объекты.

        Args:
            name (str): идентификатор модуля, строка, 
                содержащая только латинские буквы, цифры и символ подчеркивания.
                Должен быть уникальным
            bbox (tuple): массив координат, задающих прямоугольник: 
                (долгота-мин, широта-мин, долгота-макс, широта-макс)
            alias (str, optional): Отображаемое имя модуля. Defaults to "".
        """
        self.name = name
        self.bbox = bbox
        self.alias = alias

class LayerPO:
    def __init__(self, name: str, 
                       ordering: int, 
                       alias:str = "", 
                       area=None,
                       is_vector=True):
        """Слой

        Args:
            name (str): идентификатор, cм. выше
            ordering (int): положение слоя, отсчитывается сверху вниз (0 - самый верхний)
            alias (str, optional): Отображаемое имя слоя. Defaults to "".
            area (str, optional): Идентификатор области. Если указан, то
                слой будет виден только в данной области, 
                в противном случае - во всех областях.
                Defaults to None.
            is_vector (bool, optional): Тип слоя - векторный(True) или растровый(False).
                Defaults to True.
        """
        self.name = name
        self.area = area
        self.alias = alias
        self.ordering = ordering
        self.is_vector = is_vector

class FeaturePO:
    def __init__(self, name: str = "", date: datetime = None):
        self.name = name
        self.datetime = date

# векторные данные
class VectorFeaturePO(FeaturePO):
    def __init__(self, properties: dict, geometry: tuple, date: datetime = None):
        """
        Векторные пространственные данные

        Args:
            properties (dict): атрибуты, словарь примитивных типов python. Лучше не использовать вложенные словари
            geometry (tuple): список объектов векторной геометрии geos:
               https://docs.djangoproject.com/en/4.1/ref/contrib/gis/geos/#geometry-objects
            date (datetime, optional): дата и время. Defaults to None.
        """
        super().__init__(date)
        self.properties = properties
        self.geometry = geometry

class RasterFeaturePO(FeaturePO):
    def __init__(self, file, date: datetime = None):
        super().__init__(date)
        self.file = file
