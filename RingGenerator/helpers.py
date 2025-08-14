"""
Helper functions for RingGenerator
"""

from typing import Tuple, Optional
from django.contrib.gis.geos import Point
from DataExtractor.models import GISFeature


def get_coordinates_from_feature(feature: GISFeature) -> Optional[Tuple[float, float]]:
    """
    Extract latitude and longitude from a GIS feature
    
    Args:
        feature: GISFeature instance with geometry
        
    Returns:
        Tuple of (latitude, longitude) or None if geometry is invalid
    """
    if not feature.geometry:
        return None
    
    # Handle different geometry types
    if feature.geometry.geom_type == 'Point':
        # For Point geometries, extract coordinates directly
        return (feature.geometry.y, feature.geometry.x)  # lat, lon
    
    elif feature.geometry.geom_type in ['LineString', 'Polygon', 'MultiPoint', 'MultiLineString', 'MultiPolygon']:
        # For other geometry types, get the centroid
        centroid = feature.geometry.centroid
        return (centroid.y, centroid.x)  # lat, lon
    
    return None


def get_lat_lon_from_point(point: Point) -> Tuple[float, float]:
    """
    Extract latitude and longitude from a Point geometry
    
    Args:
        point: Point geometry
        
    Returns:
        Tuple of (latitude, longitude)
    """
    return (point.y, point.x)  # lat, lon


def get_point_from_lat_lon(lat: float, lon: float) -> Point:
    """
    Create a Point geometry from latitude and longitude
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        Point geometry
    """
    return Point(lon, lat)  # Point takes (x, y) which is (lon, lat)


def validate_coordinates(lat: float, lon: float) -> bool:
    """
    Validate that coordinates are within reasonable bounds
    
    Args:
        lat: Latitude
        lon: Longitude
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return -90 <= lat <= 90 and -180 <= lon <= 180


