import os
import zipfile
from django.core.management.base import BaseCommand
from DataExtractor.models import GISFeature
from django.contrib.gis.geos import GEOSGeometry
import xml.etree.ElementTree as ET

BASE_NAMES = ['fpoi', 'joint', 'landmark', 'olt', 'ont', 'ri', 'splitter']

class Command(BaseCommand):
    help = 'Extracts KMZ files and loads GIS data into the database'

    def add_arguments(self, parser):
        parser.add_argument('kmz_dir', type=str, help='Directory containing KMZ files')

    def handle(self, *args, **options):
        kmz_dir = options['kmz_dir']
        for base in BASE_NAMES:
            kmz_path = os.path.join(kmz_dir, f"{base}.kmz")
            if not os.path.exists(kmz_path):
                self.stdout.write(self.style.WARNING(f"KMZ not found: {kmz_path}"))
                continue

            with zipfile.ZipFile(kmz_path, 'r') as zf:
                kml_files = [f for f in zf.namelist() if f.endswith('.kml')]
                if not kml_files:
                    self.stdout.write(self.style.WARNING(f"No KML in {kmz_path}"))
                    continue
                kml_data = zf.read(kml_files[0]).decode('utf-8')
                root = ET.fromstring(kml_data)
                ns = {'kml': 'http://www.opengis.net/kml/2.2'}

                for placemark in root.findall('.//kml:Placemark', ns):
                    name = placemark.find('kml:name', ns)
                    desc = placemark.find('kml:description', ns)
                    geom = placemark.find('.//kml:Point', ns) or placemark.find('.//kml:Polygon', ns) or placemark.find('.//kml:LineString', ns)
                    if geom is not None:
                        geom_wkt = None
                        if geom.tag.endswith('Point'):
                            coords = geom.find('kml:coordinates', ns).text.strip()
                            lon, lat, *_ = map(float, coords.split(','))
                            geom_wkt = f'POINT({lon} {lat})'
                        elif geom.tag.endswith('LineString'):
                            coords = geom.find('kml:coordinates', ns).text.strip().split()
                            points = ', '.join(['{} {}'.format(*map(float, c.split(',')[:2])) for c in coords])
                            geom_wkt = f'LINESTRING({points})'
                        elif geom.tag.endswith('Polygon'):
                            coords = geom.find('.//kml:coordinates', ns).text.strip().split()
                            points = ', '.join(['{} {}'.format(*map(float, c.split(',')[:2])) for c in coords])
                            geom_wkt = f'POLYGON(({points}))'
                        if geom_wkt:
                            GISFeature.objects.create(
                                name=name.text if name is not None else '',
                                description=desc.text if desc is not None else '',
                                geometry=GEOSGeometry(geom_wkt, srid=4326),
                                source_file=base
                            )
                            self.stdout.write(self.style.SUCCESS(f"Imported {name.text if name is not None else 'Unnamed'} from {base}.kmz"))