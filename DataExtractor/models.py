from django.db import models

# Create your models here.
from django.contrib.gis.db import models

class GISFeature(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    geometry = models.GeometryField()
    source_file = models.CharField(max_length=50)

    def __str__(self):
        return f"{self.name} ({self.source_file})"