from .views import DrawZone
from common.models import AreaPO, LayerPO
from common.commands import CommandList
from common.schema import Schema

COMMANDS = CommandList(
    (DrawZone(),
    #other command
    )
)

SCHEMA = Schema(

 # TODO: импортирование областей из geojson / shapefile
    areas=(AreaPO(name="hibins", 
                  alias="Хибины", 
                  bbox=(32.7172, 67.3483, 34.6289,  67.97463)),

           AreaPO(name="murmansk", 
                  alias="Мурманск", 
                  bbox=(32.71728, 68.897, 34.3542, 69.37257)),

           AreaPO(name="lake", 
                  alias="Водохранилище", 
                  bbox=(30.48706, 68.2449, 32.124, 68.78)),
           #other area
           ),

    layers=(LayerPO(name="ground_layer",
                    ordering=0,
                    alias="Векторный слой 1"), 

            LayerPO(name="vector_layer_1",
                    ordering=1,
                    alias="Векторный слой 2"), 

            LayerPO(name="specific_layer",
                    area="murmansk",
                    ordering=2,
                    alias="Слой, доступный в одной области")
            # other layer
            )
)
