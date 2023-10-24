# la funcion render es utilizada en Django ara renderizar plantillas HTML y
# pasar contexto (variables) a esas plantillas.
import pdb

from django.shortcuts import render
# Librerias - Api chatgpt
from django.templatetags.static import static
from django.conf import settings
from django.http import JsonResponse
from ProyectoTikTok.config import OPENAI_API_KEY, PLAY_API_KEY, PLAY_USER, LEONARDO_API_KEY
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

# Libreria para redimensionamiento proporcional
import cv2

# Libreria para audio
import time
# Libreria para inserta musica a video
import moviepy.editor as mymovie
# Librerias para audio-video
import numpy as np  # np es una biblioteca para la computación numérica en Python
from moviepy.audio.AudioClip import AudioArrayClip  # crear un clip de audio a partir de una matriz

# Libreria video:
from moviepy.editor import AudioFileClip


# pip install moviepy
# pip install requests
# pip install Pillow
# falta uno de efectos y transiciones
# pip install numpy moviepy
# pip install opencv-python # redimensionamiento


# Funcion de vista
def index(request):
    return render(request, 'video_tiktok/index.html')


# API CHATGPT
def generate_script(request):
    # Clave API
    openai.api_key = OPENAI_API_KEY

    generated_script = ""
    if request.method == "POST":
        title = request.POST.get('videoTitle')
        script = request.POST.get('videoScript')

        tema = f'Genera un guión de 90 segundos para un video que será narrado por una voz en off, el guión debe ser' \
               f' creativo y llamativo para el escuchante. El tema del guión es: {script}. No se debe rebasar el límite' \
               f' de 90 segundos ni debe ser menos, tiene que durar exactamente 90 segundos. Tiene que ser en voz ' \
               f'pasiva y con tono informativo. no incluir ningún tipo de etiqueta, ni explicar cómo se debería ' \
               f'confomar la escena hablando de imagenes, videos, musica, transiciones entre otras cosas, únicamente ' \
               f'debe ser el guión. Luego, generar 10 prompts para Dall-e para que se generen 10 imagenes diferentes' \
               f' relacionados al tema del guión, estos prompts deben ser hecho como si los hiciera un profesional ' \
               f'que sabe usar las herramientas de generación de imagenes con inteligencia artificial para generar ' \
               f'resultados impresionantes y profesionales. La secuencia de los 10 prompts, debe ir acorde al guión, ' \
               f'deben tener especificaciones técnicas que usan los profesionales para generar imágenes. Evita usar ' \
               f'palabras como genera, crea, haz, diseña y cualquier otro verbo al inicio de los prompts.'

        tema2 = f'Genera un guión de 90 segundos para un video que será narrado por una voz en off. El guión debe ser' \
                f' creativo y llamativo para el escuchante. El tema del guión es: {script}. Debe durar exactamente 90 segundos,' \
                f' en voz pasiva y con un tono informativo. No debe incluir etiquetas ni explicaciones sobre cómo se conformará' \
                f' la escena en términos de imágenes, videos, música o transiciones; solo el contenido del guión.' \
                f' Luego, generar 15 prompts para Dall-e con el propósito de crear 15 imágenes diferentes relacionadas' \
                f' al tema del guión. La secuencia de los prompts debe ser cronológica con el guión:' \
                f' Los primeros 3 prompts deben representar el contenido de los primeros 18 segundos.' \
                f' Los prompts 4 al 6, los segundos 19 al 36. Los prompts 7 al 9, los segundos 37 a 55.' \
                f' Los prompts 10 al 12, los segundos 56 a 74. Y los prompts 13 al 15, los segundos 75 a 90. ' \
                f' Asegúrate de que cada prompt este en inglés y con los máximos detalles descriptivos de cada imagen ' \
                f' para tener resultados más profesionales' \
                f' Estos prompts deben ser elaborados como si los hiciera un profesional experto en herramientas de' \
                f' generación de imágenes con inteligencia artificial, logrando resultados impresionantes y profesionales.' \
                f' Evita usar palabras como "genera", "crea", "haz", "diseña" y cualquier otro verbo al inicio de los prompts.'

        tema3 = f'Basado en el siguiente guión, ¿lo categorizarías como "relajante" o "divertida"? ' \
                f'Genera un guión de 648 caracteres para un video que será narrado por una voz en off. El guión debe ser' \
                f' creativo y llamativo para el escuchante. El tema del guión es: {script}. Debe durar exactamente 90 segundos,' \
                f' en voz pasiva y con un tono informativo. No debe incluir etiquetas ni explicaciones sobre cómo se conformará' \
                f' la escena en términos de imágenes, videos, música o transiciones; solo el contenido del guión.' \
                f' Next, generate 15 prompts for Dall-e aiming to create 15 different images related' \
                f' to the scripts theme. The sequence of prompts should be chronological with the script:' \
                f' The first 3 prompts should represent the content of the first 18 seconds.' \
                f' Prompts 4 to 6, seconds 19 to 36. Prompts 7 to 9, seconds 37 to 55.' \
                f' Prompts 10 to 12, seconds 56 to 74. And prompts 13 to 15, seconds 75 to 90. ' \
                f' Ensure each prompt is in English and contains as much descriptive detail as possible for each image ' \
                f' to achieve professional results.' \
                f' Avoid using words like "generate", "create", "make", "design", and any other verb at the beginning of the prompts.'

        print("Longitud del prompt:", len(tema3))

        resultado = openai.Completion.create(
            engine="text-davinci-003",
            prompt=tema3,
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

        # Ahora, procesar los prompts para generar imágenes
        response = generate_images(request, image_prompts)

        return response  # Esto enviará el JsonResponse de la función `generate_images`

    else:
        return JsonResponse({'error': "Delimitador no encontrado"})


# API Leonardo
"""Solicita a Leonardo AI la generación de imágenes y devuelve el ID de la generación."""


def generate_images(request, image_prompts):
    # Inicializar una lista vacia para almacenar todas las rutas de las imágenes
    all_image_paths = []

    # Para cada prompt, generar una imagen con Leonardo
    # se obtiene tanto el índice de prompt de la lista image_prompts como el valor del prompt
    for index, prompt in enumerate(image_prompts):
        # contiene las imágenes relacionadas con un prompt particular
        individual_image_paths = []
        try:
            # Llamar a la API de Leonardo para solicitar 4 imágenes por prompt
            # Configuración de la clave API de Leonardo
            leonardo_api_key = LEONARDO_API_KEY
            url = "https://cloud.leonardo.ai/api/rest/v1/generations"
            payload = {
                "height": 1024,
                "modelId": "b820ea11-02bf-4652-97ae-9ac0cc00593d",  # UUID Leonardo Diffusion
                "prompt": prompt,
                "width": 512
            }
            headers = {
                'Authorization': f'Bearer {leonardo_api_key}',
                'Content-Type': 'application/json',
                "accept": "application/json"
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                response_data = response.json()  # Convertir la respuesta en un objeto JSON
                print(response_data)
            except requests.RequestException as e:
                return JsonResponse({"error": f"Error al hacer la solicitud: {str(e)}"}, status=500)

            # Obtenemos el 'generationId' de la respuesta
            generation_id = response_data.get('sdGenerationJob', {}).get('generationId')

            # Si tenemos un generation_id, obtenemos las imágenes
            if generation_id:
                # obtenemos las imagenes
                image_data = get_images(generation_id)
                print({"sdGenerationJob": {"generationId": generation_id}})

            else:
                return JsonResponse({"error": "Generation ID not found"}, status=400)

            # Iterar sobre las imágenes en la respuesta
            for img_index, image_url in enumerate(image_data):
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
                    # image_file.write(base64.b64decode(b64_string))
                    response = requests.get(image_url)
                    image_file.write(response.content)

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


def get_images(generation_id):
    # Esperamos 35 segundos antes de intentar obtener las imágenes
    time.sleep(35)

    url = f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}"
    leonardo_api_key = LEONARDO_API_KEY
    headers = {
        'Authorization': f'Bearer {leonardo_api_key}',
        'Content-Type': 'application/json',
        "accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    response_data = response.json()

    generated_images = response_data.get('generations_by_pk', {}).get('generated_images', [])
    image_urls = [image.get('url') for image in generated_images if image.get('url')]

    return image_urls


# Redimensionamiento proporcional
def create_thumbnail(image_path):
    # Tamaño deseado de la miniatura
    target_size = (1080, 1920)

    # Cargar imagen
    imagen = cv2.imread(image_path)

    # Conocer tamaño de la imagen original
    alto, ancho, _ = imagen.shape

    # Determinamos qué dimensión (ancho o alto) tomar como referencia para mantener la relación de aspecto.
    # Se define que el alto sea 1920 y el ancho se ajuste proporcionalmente.
    r = 1920 / alto
    dim = (int(ancho * r), 1920)

    # Redimensionar imagen
    redim = cv2.resize(imagen, dim)

    # Guardar la imagen redimensionada en el mismo directorio que la imagen original
    thumbnail_path = image_path.replace('.png', '_thumbnail.png')
    cv2.imwrite(thumbnail_path, redim)

    return thumbnail_path


# Generacion de video

def generate_video(request):
    # Obtiene la música seleccionada desde el frontend
    selected_music_filename = request.POST['music']
    print("Música recibida:", request.POST['music'])

    music_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                              selected_music_filename + '.mp3')

    try:
        image_dir = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'img', 'generated_images')

        # Filtrar solo las imágenes que contienen "_thumbnail" en su nombre
        image_filenames = sorted(
            [filename for filename in os.listdir(image_dir) if filename.endswith('.png') and "_thumbnail" in filename])

        # Agrupar las imágenes en conjuntos de 5
        # grouped_images = [image_filenames[i:i + 2] for i in range(0, len(image_filenames), 2)]
        grouped_images = [image_filenames[i:i + 4] for i in range(0, len(image_filenames), 4)]

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

        # Define la ruta del audio
        audio_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'audios',
                                  'generated_audios', 'audio.mp3')

        # Obtiene la duración del audio de voz
        voice_audio_clip = AudioFileClip(audio_path)
        voice_audio_duration = voice_audio_clip.duration

        # Calcula la duración de cada imagen en función de la duración del audio de voz
        image_duration = voice_audio_duration / len(selected_images)

        clips = []
        for idx, filename in enumerate(selected_images):
            img_path = os.path.join(image_dir, filename)
            if not os.path.exists(img_path):
                continue

            # Si es el último clip, agrega un retraso (p.ej., 3 segundos) a su duración.
            # Esto es opcional, si no quieres el retraso puedes eliminar estas líneas.
            if idx == len(selected_images) - 1:
                clip_duration = image_duration + 3
            else:
                clip_duration = image_duration

            clip = ImageClip(img_path, duration=clip_duration)
            clip = clip.fadein(1).fadeout(1)
            transition_func, *transition_args = random.choice(effects)
            clip = clip.fx(transition_func, *transition_args)
            clip = clip.fx(vfx.resize, newsize=[dim * 1.2 for dim in clip.size])
            clips.append(clip)
        '''
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
            '''

        # Concatenar y Exportar
        final_clip = concatenate_videoclips(clips, method="compose")
        video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                  'generated_videos',
                                  'video_final.mp4')
        final_clip.write_videofile(video_path, codec="libx264", fps=24)

        # ruta  donde se guardará el video con música
        video_with_music_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                             'generated_videos', 'video_music.mp4')
        # Agregar música de fondo al video original
        add_music(video_path, music_path, video_with_music_path)


        # Agregar la narración (voz) al video que ya tiene música
        audio_video(video_path, audio_path, video_path)

        # Ruta final del video con voz
        output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                   'generated_videos', 'video_audio.mp4')
        # Ruta final del video con voz y música
        final_output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                         'generated_videos', 'video_audio_music.mp4')

        # Agregar la narración (voz) al video que ya tiene música
        audio_video(video_with_music_path, audio_path, final_output_path)

        return JsonResponse({'videoPath': final_output_path})

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


# Definir las categorías y sus pistas
'''
MUSIC_CATEGORIAS = {
    "relajante": [os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                               'ES_Sleepy_Hungry_baegel.mp3')],
    "divertida": [
        os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                     'ES_Empower_Osoku.mp3'),
        os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                     'ES_Fight_Club_Cushy.mp3'),
        os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                     'ES_Neon_Lights_Neon_Dreams_Forever_Sunset.mp3'),
        os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                     'ES_Streamer_Bonkers_Beat_Club.mp3'),

        os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                     'ES_SUPRA_STRLGHT.mp3'),
    ]
}


def random_categoria(categoria):
    """Elije aleatoriamente una pista de la categoría dada."""
    return random.choice(MUSIC_CATEGORIAS[categoria])



def add_music(video_path, categoria, output_path):
    video = VideoFileClip(video_path)
    bg_music_path = random_categoria(categoria)
    bg_music = AudioFileClip(bg_music_path)

    # Si la música de fondo es más corta que el video, la repetimos
    if bg_music.duration < video.duration:
        bg_music = bg_music.fx(vfx.loop, duration=video.duration)

    # Combina el audio del video con la música de fondo
    combined_audio = CompositeAudioClip([video.audio, bg_music.volumex(0.6)])

    # Asegúra que la música no exceda la duración del video
    combined_audio = combined_audio.subclip(0, video.duration)

    video_with_bg_music = video.set_audio(combined_audio)
    video_with_bg_music.write_videofile(output_path, codec='libx264')
'''


def add_music(video_path, music_path, output_path):

    videoclip = mymovie.VideoFileClip(video_path)
    audioclip = mymovie.AudioFileClip(music_path)

    print(f"Duración del video: {videoclip.duration}")
    print(f"Duración de la música de fondo: {audioclip.duration}")

    # Ajustar el volumen (volumen x 0.5 hace que el audio suene a la mitad de su volumen original)
    audioclip = audioclip.volumex(0.05)

    # Ajustar la duración de la música al video y hacer que se desvanezca al final
    audioclip = mymovie.AudioFileClip(music_path).set_duration(videoclip.duration)

    finalclip = videoclip.set_audio(audioclip)
    finalclip.write_videofile(output_path, fps=60)

    print(f"Duración del audio combinado: {finalclip.duration}")


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

    # Si el audio es más largo que el video, agregar un efecto de desvanecimiento al video
    #  La duración del desvanecimiento será la diferencia entre la duración de la voz en off y el video.
    elif audio.duration > video.duration:
        # video_repeat_count = int(audio.duration // video.duration) + 1
        # video = concatenate_videoclips([video] * video_repeat_count)
        fade_duration = audio.duration - video.duration
        video = video.crossfadeout(fade_duration)

    video_with_audio = video.set_audio(audio)
    output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                               'generated_videos', 'video_audio.mp4')
    video_with_audio.write_videofile(output_path, codec='libx264')
    print("Audio integrado con éxito!")


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
