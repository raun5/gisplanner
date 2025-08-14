from django.core.management.base import BaseCommand
from django.contrib.gis.db.models.functions import Distance, Azimuth
from django.contrib.gis.geos import Point
from django.db.models import F
from math import radians, degrees

from DataExtractor.models import GISFeature


class Command(BaseCommand):
    help = "Perform sector sweep with multi-baseline iteration."

    # Configurable parameters
    NUM_SECTORS = 8  # Number of angular slices assuming initial angle is 45 degrees
    MAX_BASELINE_TRIES = 5 # number of iterations
    BASELINE_SHIFT_DEGREES = 10 # how much we shift up the baseline by
    MIN_POINTS_PER_RING = 7 # contstraint on the number of points that have to be in the ring
    RING_STEP_METERS = 100  
    MAX_RADIUS_METERS = 1000
    RADIUS_TOLERANCE_METERS = 5

    def handle(self, *args, **options):
        # Use centroid of all points as reference center
        centroid = GISFeature.objects.aggregate_center('geometry')
        if not centroid:
            self.stdout.write(self.style.ERROR("No GISFeature points found."))
            return

        self.stdout.write(self.style.SUCCESS(f"Centroid: {centroid.x}, {centroid.y}"))

        # Sweep sectors
        sector_angle_width = 360 / self.NUM_SECTORS
        for sector_index in range(self.NUM_SECTORS):
            start_angle = sector_index * sector_angle_width
            end_angle = start_angle + sector_angle_width

            self.stdout.write(f"\n--- Sector {sector_index+1} ({start_angle}° to {end_angle}°) ---")

            rings = self.process_sector(centroid, start_angle, end_angle)

            if rings:
                self.stdout.write(self.style.SUCCESS(f"Sector {sector_index+1}: SUCCESS ({len(rings)} rings)"))
            else:
                self.stdout.write(self.style.WARNING(f"Sector {sector_index+1}: FAILED"))

    def process_sector(self, centroid, start_angle, end_angle):
        """
        Attempt multiple baselines in the sector until constraints are met.
        """
        for attempt in range(self.MAX_BASELINE_TRIES):
            baseline_angle = start_angle + attempt * self.BASELINE_SHIFT_DEGREES
            if baseline_angle > end_angle:
                break

            self.stdout.write(f"  Attempt {attempt+1}: baseline {baseline_angle}°")

            rings = self.generate_rings(centroid, baseline_angle, start_angle, end_angle)

            if self.constraints_pass(rings):
                return rings

        return None

    def generate_rings(self, centroid, baseline_angle, start_angle, end_angle):
        """
        Generate concentric rings for a given baseline and sector using PostGIS queries.
        """
        rings = []
        radius = self.RING_STEP_METERS

        while radius <= self.MAX_RADIUS_METERS:
            qs = (
                GISFeature.objects
                .annotate(
                    dist=Distance('geometry', centroid),
                    azimuth=degrees(Azimuth(centroid, F('geometry')))
                )
                .filter(
                    dist__gte=radius - self.RADIUS_TOLERANCE_METERS,
                    dist__lte=radius + self.RADIUS_TOLERANCE_METERS,
                    azimuth__gte=start_angle,
                    azimuth__lt=end_angle
                )
            )

            ring_points = list(qs)
            if ring_points:
                rings.append(ring_points)

            radius += self.RING_STEP_METERS

        return rings

    def constraints_pass(self, rings):
        """
        Check if all generated rings meet the minimum point count.
        """
        return all(len(r) >= self.MIN_POINTS_PER_RING for r in rings)


# Add this in your GISFeature model manager
from django.db.models import Manager
from django.contrib.gis.db.models.functions import Centroid

class GISFeatureManager(Manager):
    def aggregate_center(self, field_name):
        qs = self.aggregate(center=Centroid(field_name))
        return qs["center"]

GISFeature.add_to_class('objects', GISFeatureManager())
