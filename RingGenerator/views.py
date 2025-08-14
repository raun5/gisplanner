import os
import json
from datetime import datetime
from django.shortcuts import render, get_object_or_404
from django.http import HttpResponse, Http404
from django.conf import settings
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from DataExtractor.models import GISFeature


def list_ring_analyses(request):
    """List all available ring analysis files."""
    analyses = []
    
    # Look for ring analysis files in the current working directory
    # and in the media directory if configured
    search_dirs = [os.getcwd()]
    if hasattr(settings, 'MEDIA_ROOT'):
        search_dirs.append(settings.MEDIA_ROOT)
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                if filename.startswith('ring_analysis_') and filename.endswith('.png'):
                    file_path = os.path.join(search_dir, filename)
                    file_stat = os.stat(file_path)
                    
                    # Extract block name from filename
                    block_name = filename.replace('ring_analysis_', '').replace('.png', '')
                    
                    analyses.append({
                        'filename': filename,
                        'block_name': block_name,
                        'file_path': file_path,
                        'file_size': file_stat.st_size,
                        'created_date': datetime.fromtimestamp(file_stat.st_ctime),
                        'modified_date': datetime.fromtimestamp(file_stat.st_mtime),
                        'full_path': file_path,
                    })
    
    # Sort by creation date (newest first)
    analyses.sort(key=lambda x: x['created_date'], reverse=True)
    
    context = {
        'analyses': analyses,
        'total_count': len(analyses),
        'search_directories': search_dirs,
    }
    
    return render(request, 'RingGenerator/ring_analysis_list.html', context)


def view_ring_analysis(request, filename):
    """View a specific ring analysis file."""
    # Security: ensure filename is safe
    if not filename.startswith('ring_analysis_') or not filename.endswith('.png'):
        raise Http404("Invalid filename")
    
    # Look for the file in various directories
    search_dirs = [os.getcwd()]
    if hasattr(settings, 'MEDIA_ROOT'):
        search_dirs.append(settings.MEDIA_ROOT)
    
    file_path = None
    for search_dir in search_dirs:
        potential_path = os.path.join(search_dir, filename)
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        raise Http404("Ring analysis file not found")
    
    # Extract block name from filename
    block_name = filename.replace('ring_analysis_', '').replace('.png', '')
    
    # Get file information
    file_stat = os.stat(file_path)
    
    # Try to find corresponding OLT and FPOI data
    try:
        olt_data = GISFeature.objects.filter(source_file='olt', name__iexact=block_name).first()
        fpoi_count = GISFeature.objects.filter(source_file='fpoi').count()
        olt_count = GISFeature.objects.filter(source_file='olt').count()
    except:
        olt_data = None
        fpoi_count = 0
        olt_count = 0
    
    context = {
        'filename': filename,
        'block_name': block_name,
        'file_path': file_path,
        'file_size': file_stat.st_size,
        'created_date': datetime.fromtimestamp(file_stat.st_ctime),
        'modified_date': datetime.fromtimestamp(file_stat.st_mtime),
        'file_url': f'/ring-generator/view-file/{filename}',
        'olt_data': olt_data,
        'fpoi_count': fpoi_count,
        'olt_count': olt_count,
    }
    
    return render(request, 'RingGenerator/ring_analysis_detail.html', context)


def serve_ring_analysis_file(request, filename):
    """Serve the actual ring analysis image file."""
    # Security: ensure filename is safe
    if not filename.startswith('ring_analysis_') or not filename.endswith('.png'):
        raise Http404("Invalid filename")
    
    # Look for the file in various directories
    search_dirs = [os.getcwd()]
    if hasattr(settings, 'MEDIA_ROOT'):
        search_dirs.append(settings.MEDIA_ROOT)
    
    file_path = None
    for search_dir in search_dirs:
        potential_path = os.path.join(search_dir, filename)
        if os.path.exists(potential_path):
            file_path = potential_path
            break
    
    if not file_path:
        raise Http404("Ring analysis file not found")
    
    # Serve the file
    try:
        with open(file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='image/png')
            response['Content-Disposition'] = f'inline; filename="{filename}"'
            return response
    except IOError:
        raise Http404("Error reading file")


def generate_new_analysis(request):
    """Form view to generate a new ring analysis."""
    if request.method == 'POST':
        # This would typically call the management command
        # For now, just show a message
        messages.info(request, "Ring analysis generation is done via management commands. Use: python manage.py generate_rings --block_name NAME --max_radius RADIUS --visualize --graphs")
    
    # Get available OLT blocks
    olt_blocks = GISFeature.objects.filter(source_file='olt').values_list('name', flat=True)
    fpoi_count = GISFeature.objects.filter(source_file='fpoi').count()
    
    context = {
        'olt_blocks': olt_blocks,
        'fpoi_count': fpoi_count,
        'command_example': 'python manage.py generate_rings --block_name "OLT_BLOCK_NAME" --max_radius 1000 --visualize --graphs'
    }
    
    return render(request, 'RingGenerator/generate_analysis.html', context)


def ring_generator_dashboard(request):
    """Main dashboard for ring generation functionality."""
    # Get statistics
    olt_count = GISFeature.objects.filter(source_file='olt').count()
    fpoi_count = GISFeature.objects.filter(source_file='fpoi').count()
    
    # Count existing analysis files
    analysis_count = 0
    search_dirs = [os.getcwd()]
    if hasattr(settings, 'MEDIA_ROOT'):
        search_dirs.append(settings.MEDIA_ROOT)
    
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                if filename.startswith('ring_analysis_') and filename.endswith('.png'):
                    analysis_count += 1
    
    # Get recent analyses
    recent_analyses = []
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for filename in os.listdir(search_dir):
                if filename.startswith('ring_analysis_') and filename.endswith('.png'):
                    file_path = os.path.join(search_dir, filename)
                    if os.path.exists(file_path):
                        file_stat = os.stat(file_path)
                        recent_analyses.append({
                            'filename': filename,
                            'block_name': filename.replace('ring_analysis_', '').replace('.png', ''),
                            'created_date': datetime.fromtimestamp(file_stat.st_ctime),
                        })
                        if len(recent_analyses) >= 5:  # Limit to 5 most recent
                            break
            if len(recent_analyses) >= 5:
                break
    
    # Sort by creation date
    recent_analyses.sort(key=lambda x: x['created_date'], reverse=True)
    
    context = {
        'olt_count': olt_count,
        'fpoi_count': fpoi_count,
        'analysis_count': analysis_count,
        'recent_analyses': recent_analyses,
        'has_data': olt_count > 0 and fpoi_count > 0,
    }
    
    return render(request, 'RingGenerator/dashboard.html', context)
