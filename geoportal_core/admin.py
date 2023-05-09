from django.contrib import admin
from django.db import transaction
from . import models, forms

@admin.register(models.GISModule)
class GISModuleAdmin(admin.ModelAdmin):
    form = forms.EditGISModuleForm

    def get_queryset(self, request):
        if request.user.is_superuser:
            return models.GISModule.objects.all()
        return models.GISModule.objects.filter(owner=request.user)

    def get_form(self, request, obj=None, **kwargs):
        defaults = {}
        if obj is None:
            defaults['form'] = forms.CreateGISModuleForm
        defaults.update(kwargs)
        return super().get_form(request, obj, **defaults)

    def save_model(self, request, obj, form, change):
        obj.owner = request.user
        super().save_model(request, obj, form, change)

    def delete_queryset(self, request, queryset):
        with transaction.atomic():
            for module in queryset:
                module.remove_dir()
                module.delete()

    def delete_model(self, request, obj):
        obj.remove_dir()
        super().delete_model(request, obj)


@admin.register(models.Area)
class AreaAdmin(admin.ModelAdmin):

    def get_queryset(self, request):
        if request.user.is_superuser:
            return models.Area.objects.all()
        return models.Area.objects.filter(module__owner=request.user)
    
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "module" and not request.user.is_superuser:
            kwargs["queryset"] = models.GISModule.objects.filter(owner=request.user)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)


@admin.register(models.Layer) 
class LayerAdmin(admin.ModelAdmin):
    form = forms.LayerAdminForm

    def get_queryset(self, request):
        if request.user.is_superuser:
            return models.Layer.objects.all()
        return models.Layer.objects.filter(module__owner=request.user)
    
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "module" and not request.user.is_superuser:
            kwargs["queryset"] = models.GISModule.objects.filter(owner=request.user)
        if db_field.name == "area" and not request.user.is_superuser:
            kwargs["queryset"] = models.Area.objects.filter(module__owner=request.user)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)
    
       
@admin.register(models.VectorFeature)
class VectorFeatureAdmin(admin.ModelAdmin):
    form = forms.VectorFeatureAdminForm

    def get_queryset(self, request):
        if request.user.is_superuser:
            return models.VectorFeature.objects.all()
        return models.VectorFeature.objects.filter(area__module__owner=request.user)
    
    def formfield_for_foreignkey(self, db_field, request, **kwargs):
        if db_field.name == "layer" and not request.user.is_superuser:
            kwargs["queryset"] = models.Layer.objects.filter(module__owner=request.user, layer_type='V')
        if db_field.name == "area" and not request.user.is_superuser:
            kwargs["queryset"] = models.Area.objects.filter(module__owner=request.user)
        return super().formfield_for_foreignkey(db_field, request, **kwargs)




