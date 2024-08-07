from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_file, name='upload_file'),
    path('stream_video/', views.stream_video, name='stream_video'),
]
