# la funcion render es utilizada en Django ara renderizar plantillas HTML y
# pasar contexto (variables) a esas plantillas.
from django.shortcuts import render
# Librerias - Api chatgpt
from django.templatetags.static import static
from django.conf import settings
from django.http import JsonResponse
from ProyectoTikTok.config import OPENAI_API_KEY, PLAY_API_KEY, PLAY_USER
import openai

# Librerias para descargar imagenes
import requests
from PIL import Image  # libreria pillow para miniaturas de imagenes
import os
import base64
# Librerias para generacion de video
from moviepy.editor import ImageClip, concatenate_videoclips
from moviepy.video import fx as vfx
from moviepy.video.compositing import transitions
import random
from moviepy.editor import *

# Libreria para audio
import time
# Librerias para audio-videp
import numpy as np  # np es una biblioteca para la computación numérica en Python
from moviepy.audio.AudioClip import AudioArrayClip  # crear un clip de audio a partir de una matriz


# pip install moviepy
# pip install requests
# pip install Pillow
# falta uno de efectos y transiciones
# pip install numpy moviepy


# Funcion de vista
def index(request):
    return render(request, 'video_tiktok/index.html')


# API CHATGPT
def generate_script(request):
    generated_script = ""
    if request.method == "POST":
        title = request.POST.get('videoTitle')
        script = request.POST.get('videoScript')

        # Prompt por defecto

        default_prompt = ("Genera un guión titulado 'Guion:', de 90 segundos para un video que será narrado "
                          "por una voz, en off el guión debe ser creativo y llamativo para el escuchante. El "
                          "tema del guion es:  ")

        prompt2 = (" El guión debe durar exactamente 90 segundos y estar en voz pasiva con un tono "
                   "informativo. No debe contener ninguna etiqueta, ni mencionar cómo se conformará la "
                   "escena en términos de imágenes, videos, música o transiciones. Después del guión, escribe "
                   "'Prompts para Dall-E:', y genera una lista de 10 prompts para Dall-e que permitan crear 10 "
                   "imágenes relacionadas al tema del guión. Estos prompts deben estar diseñados como si un "
                   "profesional experto en herramientas de generación de imágenes con inteligencia artificial "
                   "los hubiera hecho. Deben ser impresionantes y profesionales. La secuencia de los 10 prompts "
                   "debe coincidir con la narrativa del guión y contener especificaciones técnicas que los "
                   "profesionales usarían para generar imágenes. Evita usar palabras como genera, crea, haz, "
                   "diseña y otros verbos al inicio de los prompts.Todos los resultados de los prompts deben"
                   " tener una resolución de 1080x1920 ")

        # Clave API
        openai.api_key = OPENAI_API_KEY

        tema = default_prompt + " " + script + " " + prompt2
        print("Longitud del prompt:", len(tema))

        resultado = openai.Completion.create(
            engine="text-davinci-003",
            prompt=tema,
            max_tokens=2048,
            temperature=0.8  # indica creatividad del modelo
        )

        # Accede al texto generado en la respuesta
        generated_response = resultado.choices[0].text.strip()
        print(generated_response)

        return JsonResponse({'generated_script': generated_response})

    return JsonResponse({'error': 'Método no permitido'}, status=405)


# Procesamiento de la respuesta:
def split_prompts(request):
    generated_response = request.POST.get('editedScript')
    delimiter = "Prompts para Dall-E:"
    print(repr(generated_response))

    if delimiter in generated_response:
        # Dividir la respuesta en guion y prompts usando el delimitador
        parts = generated_response.split(delimiter, 1)

        # Procesar el guión
        script_to_show = parts[0].replace("Guion:", "").strip()
        script_to_show = script_to_show.replace("Guión:", "").strip()

        # Procesar los prompts
        image_prompts = parts[1].strip().split('\n')
        image_prompts = [prompt.strip() for prompt in image_prompts if prompt.strip() != ""]

        # Ahora, procesar los prompts para generar imágenes usando DALL·E
        # response = generate_images(image_prompts)
        response = generate_images(request, image_prompts)

        return response  # Esto enviará el JsonResponse de la función `generate_images`

    else:
        return JsonResponse({'error': "Delimitador no encontrado"})

        # Ahora, procesar los prompts para generar imágenes
        image_paths = []
        for prompt in image_prompts:
            image_path = get_image_from_dalle(prompt)  # Función que envía prompt a DALL·E y obtiene una imagen.
            image_paths.append(image_path)

    return JsonResponse({'image_prompts': image_prompts})


# API DALL-E

# Configuración de la clave API de OpenAI
openai.api_key = OPENAI_API_KEY


def generate_images(request, image_prompts):
    # Inicializar una lista vacia para almacenar todas las rutas de las imágenes
    all_image_paths = []

    # Para cada prompt, generar una imagen con DALL·E
    # se obtiene tanto el índice de prompt de la lista image_prompts como el valor del prompt
    for index, prompt in enumerate(image_prompts):
        # contiene las imágenes relacionadas con un prompt particular
        individual_image_paths = []
        try:
            # Llamar a la API de DALL·E para generar 2 imágenes por prompt
            respuesta = openai.Image.create(
                prompt=prompt,
                n=5,
                size="1024x1024",
                response_format="b64_json"
            )

            # Iterar sobre las imágenes en la respuesta
            for img_index, imagen in enumerate(respuesta['data']):
                # Obtener la cadena base64 de la imagen Json
                b64_string = imagen['b64_json']

                # Definir el nombre de la imagen basado en el indice del prompt y el indice de la imagen individual
                image_file_name = f"imagen_prompt_{index}_img_{img_index}.png"

                # Definir la ruta donde se guardará la imagen
                base_dir = settings.BASE_DIR  # obtenemos el directorio base del proyecto
                # Construimos el camino al directorio donde se guardarán las imágenes generadas.
                static_dir = os.path.join(base_dir, 'video_tiktok', 'static', 'video_tiktok', 'img', 'generated_images')
                # Combinamos el directorio estático con el nombre de archivo de la imagen para obtener el camino
                # completo donde guardaremos la imagen.
                image_file_path = os.path.join(static_dir, image_file_name)

                # Asegurarse de que el directorio exista. Si no existe, se creará
                os.makedirs(static_dir, exist_ok=True)

                # Decodificar la cadena base64 y guardar la imagen original
                with open(image_file_path, "wb") as image_file:
                    image_file.write(base64.b64decode(b64_string))

                # Llamamos a la funcion create_thumbnail para crear la miniatura
                thumbnail_path = create_thumbnail(image_file_path)

                # obtenemos la ruta relativa de thumbnail_path (miniatura)
                rel_thumbnail_path = os.path.relpath(thumbnail_path, os.path.join(settings.BASE_DIR, 'static'))
                # obtenemos la ruta relativa de image_file_path (imagen original)
                rel_image_path = os.path.relpath(image_file_path, os.path.join(settings.BASE_DIR, 'static'))

                # Convertimos las rutas relativas a URLs usando la función static de Django.
                thumbnail_url = static(rel_thumbnail_path)
                image_url = static(rel_image_path)

                # Imprimir las URLs en la consola del servidor
                print("Image URL:", image_url)
                print("Thumbnail URL:", thumbnail_url)

                # Agregar las URLs de la imagen y la miniatura a la lista individual_image_paths
                individual_image_paths.append({"image": image_url, "thumbnail": thumbnail_url})

            # Agregar todas las imágenes generadas para este prompt a la lista general
            all_image_paths.append(individual_image_paths)
            request.session['all_image_paths'] = all_image_paths  # Guardar en la sesión

        except Exception as e:
            # Imprimir la excepción
            print(f"Error al generar imagen para el prompt: {prompt}")
            print(e)
            # Devolver un mensaje de error
            return JsonResponse({"error": str(e)})

    # Devolver todas las rutas de las imágenes en la respuesta JSON
    return JsonResponse({"imagePaths": all_image_paths})


# Creamos la miniatura
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
    try:
        image_dir = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'img', 'generated_images')

        # Filtrar solo las imágenes que contienen "_thumbnail" en su nombre
        image_filenames = sorted(
            [filename for filename in os.listdir(image_dir) if filename.endswith('.png') and "_thumbnail" in filename])

        # Agrupar las imágenes en conjuntos de 2 (puedes cambiar esto a 5 en el futuro)
        # grouped_images = [image_filenames[i:i + 2] for i in range(0, len(image_filenames), 2)]
        grouped_images = [image_filenames[i:i + 5] for i in range(0, len(image_filenames), 5)]

        # Selecciona una sola imagen aleatoriamente de las generadas por cada prompt
        # selected_images = [random.choice(images_list) for images_list in image_filenames]

        # Luego, selecciona una imagen aleatoriamente de cada grupo
        selected_images = [random.choice(group) for group in grouped_images]

        # Crear una lista de funciones y argumentos
        effects = [
            # Aplica una transición de deslizamiento desde la derecha a la imagen durante un segundo
            (transitions.slide_in, 1, 'right'),
            #  Aplica una transición de deslizamiento hacia la izquierda a la imagen durante un segundo.
            (transitions.slide_out, 1, 'left'),
            # Aplica una transición de desvanecimiento (aparece gradualmente) a la imagen durante un segundo.
            (transitions.fadein, 1),
            # desaparece gradualmente
            (transitions.fadeout, 1),
            # Invierte los colores de la imagen.
            # (vfx.invert_colors,),
            # Aplica un efecto de pintura a la imagen durante un segundo.
            (vfx.painting, 1)
        ]

        clips = []
        for filename in selected_images:  # Ahora estamos iterando sobre las imágenes seleccionadas
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                continue  # Si el archivo no existe, continúa con el siguiente
            clip = ImageClip(img_path, duration=5)
            # # Aplicar efecto de desvanecimiento (1s fadein y 1s fadeout)
            clip = clip.fadein(1).fadeout(1)
            # Efectos y Transiciones
            transition_func, *transition_args = random.choice(effects)
            clip = clip.fx(transition_func, *transition_args)
            # Aplicar "zoom" - cambiar tamaño
            clip = clip.fx(vfx.resize, newsize=[dim * 1.2 for dim in clip.size])
            clips.append(clip)

        # Concatenar y Exportar
        final_clip = concatenate_videoclips(clips, method="compose")
        video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                  'generated_videos',
                                  'video_final.mp4')
        final_clip.write_videofile(video_path, codec="libx264", fps=24)

        # Define la ruta del audio
        audio_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'audios',
                                  'generated_audios', 'audio.mp3')

        # Una vez que el video está listo, integrarlo con el audio
        audio_video(video_path, audio_path, video_path)

        # return JsonResponse({'videoPath': video_path})
        output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                   'generated_videos', 'video_audio.mp4')
        return JsonResponse({'videoPath': output_path})

    except Exception as e:
        return JsonResponse({"error": str(e)})


# Convertir texto a voz

def texto_a_voz(request):
    texto = request.POST.get('texto')
    selected_voice = request.POST.get('voice')

    # Procesar para extraer el guion
    delimiter = "Prompts para Dall-E:"
    if delimiter in texto:
        parts = texto.split(delimiter, 1)
        script = parts[0].replace("Guion:", "").strip()
        script = script.replace("Guión:", "").strip()

    API_KEY = PLAY_API_KEY
    HEADERS = {
        'Authorization': 'Bearer ' + API_KEY,
        'Content-Type': 'application/json',
        "accept": "application/json",
        "X-User-ID": PLAY_USER
    }
    url = "https://play.ht/api/v1/convert"

    payload = {
        "content": [script],
        "voice": selected_voice
    }

    response = requests.post(url, json=payload, headers=HEADERS)
    data = response.json()  # Convertir la respuesta en un objeto JSON
    transcriptionId = data.get('transcriptionId')

    print(data)

    if transcriptionId:
        generate_voz(transcriptionId)
        return JsonResponse({"status": "success", "transcriptionId": transcriptionId})
    else:
        return JsonResponse({"status": "error"}, status=400)


def generate_voz(transcriptionId):
    url = f"https://play.ht/api/v1/articleStatus?transcriptionId={transcriptionId}"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer 28c8e53694d047d6bf008eea60c5ff6f",
        "X-User-ID": PLAY_USER
    }

    for _ in range(10):  # Intentar hasta 10 veces
        response = requests.get(url, headers=headers)
        data = response.json()

        # Si la conversión ha finalizado
        if data.get('converted'):
            audio_url = data.get('audioUrl')
            if audio_url:
                audio_response = requests.get(audio_url)
                audio_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'audios',
                                          'generated_audios', 'audio.mp3')
                with open(audio_path, 'wb') as f:
                    f.write(audio_response.content)
                print("Audio descargado con éxito.")
                return JsonResponse({"status": "success", "transcriptionId": transcriptionId, "audioPath": audio_path})
        else:
            time.sleep(5)  # Esperar 5 segundos antes del próximo intento

    print("No se pudo obtener la URL del audio después de varios intentos.")
    return JsonResponse(
        {"status": "error", "message": "No se pudo obtener la URL del audio después de varios intentos."}, status=400)


def audio_video(video_path, audio_path, output_path):
    print("Cargando clips de video y audio...")
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)

    print(f"Duración del video: {video.duration}")
    print(f"Duración del audio: {audio.duration}")

    # Si el audio es más corto que el video, añadir silencio al final del audio
    if audio.duration < video.duration:
        silence_duration = video.duration - audio.duration
        silence = AudioArrayClip(np.array([[0, 0]]), fps=44100).set_duration(
            silence_duration)  # Genera un clip de silencio
        audio = concatenate_audioclips([audio, silence])

    # Si el audio es más largo que el video, repetir el video hasta que alcance la duración del audio
    elif audio.duration > video.duration:
        video_repeat_count = int(audio.duration // video.duration) + 1
        video = concatenate_videoclips([video] * video_repeat_count)

    video_with_audio = video.set_audio(audio)
    output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                               'generated_videos', 'video_audio.mp4')
    video_with_audio.write_videofile(output_path, codec='libx264')


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
