from django.urls import path
from . import views

app_name = 'ring_generator'

urlpatterns = [
    # Main dashboard
    path('', views.ring_generator_dashboard, name='dashboard'),
    
    # List all ring analyses
    path('analyses/', views.list_ring_analyses, name='analysis_list'),
    
    # View specific ring analysis
    path('analysis/<str:filename>/', views.view_ring_analysis, name='analysis_detail'),
    
    # Serve the actual image file
    path('view-file/<str:filename>/', views.serve_ring_analysis_file, name='view_file'),
    
    # Generate new analysis
    path('generate/', views.generate_new_analysis, name='generate_analysis'),
]
