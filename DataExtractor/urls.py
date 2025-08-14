from django.urls import path
from . import views

urlpatterns = [
    path('', views.map_view, name='map_view'),
    path('features/', views.features_api, name='features_api'),
    path('features/<int:feature_id>/', views.feature_detail, name='feature_detail'),
]


