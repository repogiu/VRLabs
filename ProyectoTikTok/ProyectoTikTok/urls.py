"""
URL configuration for ProyectoTikTok project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from video_tiktok.views import index
from video_tiktok import views
from django.conf import settings  # Importar settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('generate_script/', views.generate_script, name='generate_script'),
    path('generate_images/', views.generate_images, name='generate_images'),
    path('generate_video/', views.generate_video, name='generate_video'),
    path('delete_video/', views.delete_video, name='delete_video'),

]

# Esto es solo necesario cuando DEBUG = True:
# Django agregará las rutas para servir los archivos estáticos automáticamente
if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()
