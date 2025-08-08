from django.contrib import admin
from .models import GISFeature

# Register your models here.

@admin.register(GISFeature)
class GISFeatureAdmin(admin.ModelAdmin):
    list_display = ('name', 'source_file')
