# la funcion render es utilizada en Django ara renderizar plantillas HTML y
# pasar contexto (variables) a esas plantillas.
from django.shortcuts import render
# Librerias - Api chatgpt
from django.templatetags.static import static
from django.conf import settings
from django.http import JsonResponse
from ProyectoTikTok.config import OPENAI_API_KEY
import openai
# Librerias para descargar imagenes
import requests
from PIL import Image  # libreria pillow para miniaturas de imagenes
import os
import base64
# Librerias para generacion de video
from moviepy.editor import concatenate_videoclips, ImageClip


# pip install moviepy
# pip install requests
# pip install Pillow


# Funcion de vista
def index(request):
    return render(request, 'video_tiktok/index.html')


# API CHATGPT
def generate_script(request):
    generated_script = ""
    if request.method == "POST":
        title = request.POST.get('videoTitle')
        script = request.POST.get('videoScript')

        # Aquí llamarias a la API de OpenAI con los datos recibidos
        openai.api_key = OPENAI_API_KEY

        tema = title + " " + script
        print("Longitud del prompt:", len(tema))

        resultado = openai.Completion.create(
            engine="text-davinci-003",
            prompt=tema,
            max_tokens=1000,
            temperature=0.8  # indica creatividad del modelo
        )

        # Accede al texto generado en la respuesta
        generated_script = resultado.choices[0].text.strip()

    return JsonResponse({'generated_script': generated_script})


# API DALL-E

# Configuración de la clave API de OpenAI
openai.api_key = OPENAI_API_KEY


def generate_images(request):
    # Comprobar si el método de la solicitud es POST
    if request.method == "POST":
        # Obtener el script editado desde la solicitud POST
        edited_script = request.POST.get('editedScript')

        # Verificar si el script editado está presente
        if edited_script:
            try:
                # Llamar a la API de DALL·E para generar imágenes
                respuesta = openai.Image.create(
                    prompt=edited_script,
                    n=4,
                    size="1024x1024",
                    response_format="b64_json"  # Solicitar respuesta en formato binario JSON
                )

                # Inicializar la lista para almacenar las rutas de las imágenes
                image_paths = []

                # Iterar sobre las imágenes en la respuesta
                for i, imagen in enumerate(respuesta['data']):
                    # Obtener la cadena base64 de la imagen
                    b64_string = imagen['b64_json']

                    # Definir el nombre del archivo de la imagen
                    image_file_name = f"imagen_{i}.png"

                    # Definir la ruta donde se guardará la imagen
                    # base_dir = os.path.dirname(os.path.abspath(__file__))  # Directorio actual del archivo
                    base_dir = settings.BASE_DIR  # Directorio base del proyecto
                    static_dir = os.path.join(base_dir, 'video_tiktok', 'static', 'video_tiktok', 'img',
                                              'generated_images')

                    image_file_path = os.path.join("video_tiktok/static/video_tiktok/img/generated_images",
                                                   image_file_name)
                    image_file_path = os.path.join(static_dir, image_file_name)

                    # Asegurarse de que el directorio exista
                    os.makedirs(static_dir, exist_ok=True)

                    # Definir la ruta completa del archivo de imagen
                    # image_file_path = os.path.join(static_dir, image_file_name)

                    # Decodificar la cadena base64 y guardar la imagen
                    with open(image_file_path, "wb") as image_file:
                        image_file.write(base64.b64decode(b64_string))

                    # Crear la miniatura
                    thumbnail_path = create_thumbnail(image_file_path)

                    # Obtener la ruta relativa de la miniatura e imagen
                    rel_thumbnail_path = os.path.relpath(thumbnail_path, os.path.join(settings.BASE_DIR, 'static'))
                    rel_image_path = os.path.relpath(image_file_path, os.path.join(settings.BASE_DIR, 'static'))

                    # Crear la URL de la miniatura utilizando la etiqueta static de Django
                    # thumbnail_url = static(thumbnail_path)
                    thumbnail_url = static(rel_thumbnail_path)

                    # Crear la URL de la imagen utilizando la etiqueta static de Django

                    # image_url = os.path.join("/static/video_tiktok/img/generated_images",
                    #                         image_file_name)
                    # image_url = static(os.path.join("video_tiktok/img/generated_images",
                    #                                image_file_name))

                    # Crear la URL de la imagen utilizando la etiqueta static de Django
                    image_url = static(rel_image_path)

                    # Agregar la ruta de la imagen a la lista
                    # image_paths.append(image_file_path)
                    # image_paths.append(image_url)

                    # Agregar la ruta de la miniatura a la lista
                    # image_paths.append(thumbnail_url)

                    # Imprimir las URLs en la consola del servidor
                    print("Image URL:", image_url)
                    print("Thumbnail URL:", thumbnail_url)

                    # Agregar las URLs de la imagen y la miniatura a la lista
                    image_paths.append({"image": image_url, "thumbnail": thumbnail_url})

                # Devolver las rutas de las imágenes en la respuesta JSON
                return JsonResponse({"imagePaths": image_paths})

            except Exception as e:
                # Imprimir la excepción
                print(e)
                # Devolver un mensaje de error
                return JsonResponse({"error": str(e)})

    # Devolver un mensaje de error si el método de la solicitud no es POST
    return JsonResponse({"error": "Método no permitido"}, status=405)


''''
# Miniaturas de las imagenes (thumbnails)

def create_thumbnail(image_path):
    # Tamaño deseado de la miniatura
    #size = (128, 128)
    size = (1080, 1920)

    # Abrir la imagen original
    original_image = Image.open(image_path)

    # Crear la miniatura
    original_image.thumbnail(size)

    # Guardar la miniatura en el mismo directorio que la imagen original
    # con un sufijo "_thumbnail" en el nombre del archivo
    thumbnail_path = image_path.replace('.png', '_thumbnail.png')
    original_image.save(thumbnail_path)

    return thumbnail_path
'''


def create_thumbnail(image_path):
    # Tamaño deseado de la miniatura
    target_size = (1080, 1920)
    target_aspect_ratio = target_size[0] / target_size[1]

    # Abrir la imagen original
    original_image = Image.open(image_path)
    orig_width, orig_height = original_image.size
    orig_aspect_ratio = orig_width / orig_height

    # Determinar dimensiones del recorte para mantener la relación de aspecto
    if orig_aspect_ratio > target_aspect_ratio:
        # Imagen original más ancha que el objetivo
        new_width = int(orig_height * target_aspect_ratio)
        offset = (orig_width - new_width) / 2
        crop_box = (offset, 0, orig_width - offset, orig_height)
    else:
        # Imagen original más alta que el objetivo
        new_height = int(orig_width / target_aspect_ratio)
        offset = (orig_height - new_height) / 2
        crop_box = (0, offset, orig_width, orig_height - offset)

    cropped_image = original_image.crop(crop_box)
    resized_image = cropped_image.resize(target_size)

    # Guardar la imagen redimensionada en el mismo directorio que la imagen original
    thumbnail_path = image_path.replace('.png', '_thumbnail.png')
    resized_image.save(thumbnail_path)

    return thumbnail_path


# Generacion de video

def generate_video(request):
    # Obtén la ruta del directorio donde se almacenan las imágenes generadas
    image_dir = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'img', 'generated_images')

    # Crea una lista de las rutas de las imágenes en el directorio
    # images = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir) if filename.endswith('.png')]
    imagenes = [os.path.join(image_dir, archivo) for archivo in os.listdir(image_dir)]

    # Ordena la lista de imágenes si es necesario
    # images.sort()

    # Crea ImageClips individuales para cada imagen y especifica su duración
    # clips = [ImageClip(image).set_duration(2) for image in images]

    clip1 = ImageClip(os.path.join(image_dir, "imagen_0_thumbnail.png")).set_duration(4)
    clip2 = ImageClip(os.path.join(image_dir, "imagen_1_thumbnail.png")).set_duration(8)
    clip3 = ImageClip(os.path.join(image_dir, "imagen_2_thumbnail.png")).set_duration(12)
    clip4 = ImageClip(os.path.join(image_dir, "imagen_3_thumbnail.png")).set_duration(16)

    # Concatena los clips
    # final_clip = concatenate_videoclips(clips)
    final_clip = concatenate_videoclips([clip1, clip2, clip3, clip4])

    # Define la ruta donde se guardará el video final
    video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos', 'generated_videos',
                              'video_final.mp4')

    # Guarda el video final en un archivo
    final_clip.write_videofile(video_path, codec="libx264", fps=24)

    return JsonResponse({'videoPath': video_path})


# Eliminar video
def delete_video(request):
    # video_path = request.POST.get('video_path')
    video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos', 'generated_videos',
                              'video_final.mp4')
    absolute_path = os.path.join(settings.BASE_DIR, video_path)

    print(f"Trying to delete: {absolute_path}")  # Debug

    if os.path.exists(absolute_path):
        os.remove(absolute_path)
        print("Video deleted successfully!")  # Debug
        return JsonResponse({'success': True})
    else:
        print("Video not found!")  # Debug
        return JsonResponse({'success': False})
