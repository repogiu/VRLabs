# la funcion render es utilizada en Django ara renderizar plantillas HTML y
# pasar contexto (variables) a esas plantillas.
from django.shortcuts import render


# Funcion de vista
def index(request):
    return render(request, 'video_tiktok/index.html')
