from django.db import models

# Create your models here.

class UploadedFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)


class ProcessedImage(models.Model):
    image = models.ImageField(upload_to='processed_images/')
    uploaded_file = models.ForeignKey(UploadedFile, on_delete=models.CASCADE)