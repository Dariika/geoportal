from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('map/<str:module_name>/', views.show_map, name="map"),
    path('select_module/', views.module_list, name="module_select"),
    path('modules/', views.ModuleListView.as_view()),
    path('modules/<str:module_name>/commands/', views.get_commands),
    path('modules/<str:module_name>/commands/<str:command_name>/', views.run_command, name="run_command"),
    path('modules/<str:module_name>/areas/', views.AreaListView.as_view()),
    path('modules/<str:module_name>/areas/<str:area_name>/', views.AreaDetailView.as_view()),
    path('modules/<str:module_name>/areas/<str:area_name>/layers/', views.LayerListView.as_view()),
    path('modules/<str:module_name>/areas/<str:area_name>/layers/<str:layer_name>/', views.LayerContentListView.as_view()),
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
