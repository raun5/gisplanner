from django.shortcuts import render
from django.http import JsonResponse
from .models import GISFeature
import json

# Create your views here.

def map_view(request):
    """
    Display all GIS features on an interactive map
    """
    features = GISFeature.objects.all()
    
    # Convert to GeoJSON for Leaflet (ensure WGS84/EPSG:4326)
    geojson_data = []
    for feature in features:
        if feature.geometry:
            # Convert geometry to GeoJSON format
            geom_4326 = feature.geometry.clone()
            try:
                geom_4326.transform(4326)
            except Exception:
                # If transform fails, fall back to original geometry
                pass
            geojson_feature = {
                'type': 'Feature',
                'geometry': json.loads(geom_4326.geojson),
                'properties': {
                    'name': feature.name,
                    'description': feature.description or '',
                    'source_file': feature.source_file,
                    'id': feature.id
                }
            }
            geojson_data.append(geojson_feature)
    
    context = {
        'features': features,
        'geojson_data': json.dumps(geojson_data),
        'feature_count': features.count()
    }
    
    return render(request, 'DataExtractor/map.html', context)

def feature_detail(request, feature_id):
    """
    Get detailed information about a specific feature
    """
    try:
        feature = GISFeature.objects.get(id=feature_id)
        # Ensure geometry is WGS84 for detail response
        geom_4326 = feature.geometry.clone() if feature.geometry else None
        try:
            if geom_4326 is not None:
                geom_4326.transform(4326)
        except Exception:
            pass

        data = {
            'id': feature.id,
            'name': feature.name,
            'description': feature.description,
            'source_file': feature.source_file,
            'geometry_type': geom_4326.geom_type if geom_4326 is not None else None,
            'coordinates': json.loads(geom_4326.geojson) if geom_4326 is not None else None
        }
        return JsonResponse(data)
    except GISFeature.DoesNotExist:
        return JsonResponse({'error': 'Feature not found'}, status=404)

def features_api(request):
    """
    API endpoint to get all features as GeoJSON
    """
    features = GISFeature.objects.all()
    
    geojson_data = {
        'type': 'FeatureCollection',
        'features': []
    }
    
    for feature in features:
        if feature.geometry:
            geom_4326 = feature.geometry.clone()
            try:
                geom_4326.transform(4326)
            except Exception:
                pass

            geojson_feature = {
                'type': 'Feature',
                'geometry': json.loads(geom_4326.geojson),
                'properties': {
                    'name': feature.name,
                    'description': feature.description or '',
                    'source_file': feature.source_file,
                    'id': feature.id
                }
            }
            geojson_data['features'].append(geojson_feature)
    
    return JsonResponse(geojson_data)
