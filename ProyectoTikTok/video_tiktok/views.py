# la funcion render es utilizada en Django ara renderizar plantillas HTML y
# pasar contexto (variables) a esas plantillas.
from django.shortcuts import render

import pdb
# Librerias para whisper
import shutil
import warnings
import whisper  # pip install -U openai-whisper

# Librerias para los subtitulos
import pysrt  # pip install pysrt
from pysrt import SubRipTime

# Convierte srt a ass: aplica estilos al archivo ass
import pysubs2  # pip install pysubs2
import datetime

# Correccion subtitulos
import srt  # pip install srt

# Librerias - Api chatgpt
from django.templatetags.static import static
from django.conf import settings
from django.http import JsonResponse
from ProyectoTikTok.config import OPENAI_API_KEY, PLAY_API_KEY, PLAY_USER, LEONARDO_API_KEY
import openai

# Efecto zoom in/out, redimensionamiento proporcional
from PIL import Image  # pip install Pillow
import cv2  # pip install opencv-python-headless
from django.conf import settings
import subprocess
from django.core.management import call_command

# Libreria de expresion regular
import re

# Libreria para bajar volumen
from pydub import AudioSegment  # pip install pydub

# Librerias para descargar imagenes
import requests  # pip install requests
import os
import base64

# Libreria para combinar audio y musica
import moviepy.editor as mpe  # pip install moviepy

# Librerias para generacion de video
from moviepy.video import fx as vfx
from moviepy.video.compositing import transitions
from moviepy.video.VideoClip import ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip  # combinar audio y musica
from moviepy.editor import transfx, concatenate_videoclips
from moviepy.video.fx import rotate
import random

# Libreria para audio
import time

# Libreria para inserta musica a video
import moviepy.editor as mymovie

# Librerias para audio-video
import numpy as np  # pip install numpy moviepy, np es una biblioteca para la computación numérica en Python
from moviepy.audio.AudioClip import AudioArrayClip  # crear un clip de audio a partir de una matriz
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_audioclips


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

        # f'Basado en el siguiente guión, ¿lo categorizarías como "relajante" o "divertida"? ' \
        ejemplo = f'El cambio climático es uno de los mayores desafíos que enfrenta la humanidad en el siglo XXI. ' \
                  f'Pero, ¿qué es exactamente el cambio climático y cómo nos afecta? A continuación, te presentamos ' \
                  f'un breve video que te explicará este fenómeno y sus consecuencias.El cambio climático se refiere ' \
                  f'a las variaciones a largo plazo del clima en la Tierra, causadas principalmente por las actividades' \
                  f' humanas que emiten gases de efecto invernadero, como el dióxido de carbono, el metano o el óxido ' \
                  f'nitroso. Estos gases atrapan el calor del sol en la atmósfera, provocando un aumento de la ' \
                  f'temperatura global. El cambio climático tiene múltiples efectos negativos para el planeta y para la' \
                  f' vida. Algunos de estos efectos son: El aumento del nivel del mar, debido al deshielo de los polos ' \
                  f'y los glaciares, lo que amenaza a las zonas costeras y a las islas. La alteración de los ecosistemas,' \
                  f' debido a la pérdida de biodiversidad, la desertificación, la deforestación o la invasión de especies' \
                  f' exóticas. La intensificación de los fenómenos meteorológicos extremos, como sequías, inundaciones, ' \
                  f'olas de calor, huracanes o incendios forestales. La afectación de la salud humana, debido al ' \
                  f'incremento de enfermedades infecciosas, respiratorias o alérgicas, así como al estrés térmico o ' \
                  f'la desnutrición. La generación de conflictos sociales, debido a la escasez de recursos naturales, ' \
                  f'como el agua o la comida, así como al desplazamiento de poblaciones o a la violencia. En resumen, ' \
                  f'el cambio climático es un problema grave que nos afecta a todos y que requiere una acción conjunta ' \
                  f'y responsable. Si queremos preservar nuestro planeta y nuestro futuro, debemos actuar ahora. ' \
                  f'Si te gustó este video, no olvides compartirlo y dejar tus comentarios. ¡Hasta la próxima!'

        tema = f'Genera un guión de destinado a una narración de voz en off para un video sobre el tema: {script}. ' \
               f'Este guión debe tener una duración de aproximadamente 90 segundos cuando se lee en voz alta.' \
               f'((((Asegurate que el guión tenga una longitud exacta de 1742 caracteres, incluyendo espacios y ' \
               f'signos de puntuación. Importante!)))). El tono del guión debe ser creativo, llamativo y fluido, con un flujo continuo' \
               f' y sin interrupciones.' \
               f'El enfoque debe estar únicamente en el contenido narrativo. A continuación, te proporciono un ejemplo' \
               f'para que tengas una mejor idea de lo que busco: {ejemplo}.' \
               f'Evita usar incluir indicaciones escénicas, etiquetas, "marcas de tiempo" "indicaciones de Voz en off"' \
               f'o similares, detalles sobre la visualización del video, como música, transiciones o imágenes al inicio del guion.'

        print("Longitud del tema:", len(tema))
        print("Longitud del ejemplo:", len(ejemplo))

        max_retries = 3  # Número máximo de intentos
        retry_count = 0

        while retry_count < max_retries:
            try:
                resultado = openai.Completion.create(
                    engine="text-davinci-003",
                    # model="gpt-3.5-turbo-instruct",
                    prompt=tema,
                    # max_tokens=2097 - len(tema),
                    max_tokens=2049,
                    temperature=0.8
                )

                generated_response = resultado.choices[0].text.strip()
                print(generated_response)
                print("Longitud del guion generado:", len(generated_response))

                return JsonResponse({'generated_script': generated_response})

            except requests.exceptions.ConnectionError:
                retry_count += 1
                print(f"Error de conexión. Reintentando {retry_count}/{max_retries}...")

            except Exception as e:
                print(f"Error: {e}")
                return JsonResponse({'error': 'Error al generar el guión'}, status=500)

        return JsonResponse({'error': 'Se agotaron los intentos de conexión'}, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)


def generate_prompt(request):
    # Clave API
    openai.api_key = OPENAI_API_KEY

    guion = request.POST.get('editedScript')

    ejemplo = f'Caballero medieval, "impresionante fotografía ultrarrealista premiada. Rostro, ojos, iris y pupila en' \
              f'simetría e hiperdetalllados. Manos perfectas". El caballero debe sostener una espada en su mano' \
              f' derecha y mirar a la cámara. El fondo debería ser un campo de batalla con humo, fuego y cadáveres. ' \
              f'La imagen debe tener una atmósfera oscura y lúgubre. Se quita el casco del caballero y se puede ver su' \
              f' rostro valiente y orgulloso. Tiene cabello rubio y ojos azules. Su mirada es firme y desafiante. '

    prompt = f'El tema del guión es: {guion}. El guion esta programado para que dure exactamente 90 segundos. ' \
             f'Tener en cuenta 150 palabras suele ser leído en aproximadamente 1 minuto. A partir de esa información' \
             f' Generar 15 prompts con el propósito de crear 15 imágenes diferentes relacionadas' \
             f' al tema del guión. La secuencia de los prompts debe ser cronológica con el guión:' \
             f' Los primeros 3 prompts deben representar el contenido de los primeros 18 segundos.' \
             f' Los prompts 4 al 6, los segundos 19 al 36. Los prompts 7 al 9, los segundos 37 a 55.' \
             f' Los prompts 10 al 12, los segundos 56 a 74. Y los prompts 13 al 15, los segundos 75 a 90. ' \
             f' Asegúrate de que cada prompt incluya detalles minuciosos y específicos de cada imagen, abordando ' \
             f'características como el vestuario, expresiones faciales, color, ambiente y contexto. Siga este ejemplo: {ejemplo} ' \
             f' Generar descripciones con escenas individuales o con un número maximo de 2 personas, evitando' \
             f'etiquetas, texto, pancartas, carteles, señales en la descripción de los prompts.' \
             f'Los detalles son esenciales para obtener imágenes realistas y de alta calidad. Estos prompts deben ser' \
             f'elaborados como si los hiciera un profesional experto en herramientas de generación de imágenes con ' \
             f'inteligencia artificial, logrando resultados impresionantes y profesionales.' \
             f'Evita usar palabras como "genera", "crea", "haz", "diseña" y cualquier otro verbo al inicio de los prompts.'

    solicitud = f'Basado en el tema del guión: {guion} programado para una duración de 90 segundos, se requiere generar ' \
                f'una lista numérica de 15 prompts detalladas para imágenes relacionadas al tema. Estas descripciones deben ser ' \
                f'cronológicas de acuerdo al guión, dividiéndose de la siguiente manera: los primeros 3 prompts para ' \
                f'los primeros 18 segundos, los siguientes 3 para los segundos 19 al 36, los siguientes 3 para los ' \
                f'segundos 37 al 55, los siguientes 3 para los segundos 56 al 74 y los últimos 3 para los segundos ' \
                f'75 al 90. Las descripciones deben ser similares en estilo y calidad al siguiente ejemplo: {ejemplo}.' \
                f'Los detalles son esenciales para obtener imágenes realistas y de alta calidad. Estos prompts deben ser' \
                f'elaborados como si los hiciera un profesional experto en herramientas de generación de imágenes con ' \
                f'inteligencia artificial, logrando resultados impresionantes y profesionales.' \
                f'Evita usar palabras como "genera", "crea", "haz", "diseña" y cualquier otro verbo al inicio de los prompts.' \
                f'Al final de cada prompt agregar lo siguiente: "Impresionante fotografía ultrarrealista premiada. Sin texto."' \
                f'Solo si hay personas, hombre, mujer, niños, personajes, agregar al final del prompt: "Rostro, ojos, iris y pupila ' \
                f'en simetría e hiperdetallados. Manos perfectas."'

    print("Longitud del prompt:", len(prompt))

    max_retries = 3  # Número máximo de intentos
    retry_count = 0

    while retry_count < max_retries:
        try:
            resultado = openai.Completion.create(
                # model="gpt-3.5-turbo-instruct",
                engine="text-davinci-003",
                prompt=solicitud,
                max_tokens=2049,
                temperature=0.8,  # indica creatividad del modelo
                timeout=10  # tiempo de espera máximo en segundos
            )

            # Accede al texto generado en la respuesta
            generated_response = resultado.choices[0].text.strip()
            print(generated_response)



            # Llamamos a la funcion split_prompts
            generated_response = split_prompts(request, generated_response)
            return generated_response

            # return JsonResponse({'generated_prompt': generated_response})
        except requests.exceptions.ConnectionError:
            retry_count += 1
            print(f"Error de conexión. Reintentando {retry_count}/{max_retries}...")
            # Esperar un tiempo exponencial entre cada reintento
            time.sleep(2 ** retry_count)
        except openai.error.OpenAIError as e:
            # Comprobar si el error es por sobrecarga de la API
            if e.status_code == 429:
                # Obtener el tiempo de espera que indica la API, por defecto 0
                retry_after = int(e.headers.get("Retry-After", 0))
                print(f"API sobrecargada. Esperando {retry_after} segundos...")
                # Esperar el tiempo indicado
                time.sleep(retry_after)
                # Reintentar la solicitud
                continue
            else:
                # Otros tipos de errores
                print(f"Error: {e}")
                return JsonResponse({'error': 'Error al generar el guión'}, status=500)
        except Exception as e:
            # Otros tipos de excepciones
            print(f"Error: {e}")
            return JsonResponse({'error': 'Error al generar el guión'}, status=500)

    return JsonResponse({'error': 'Se agotaron los intentos de conexión'}, status=500)


# Procesamiento de la respuesta:
def split_prompts(request, generated_response):
    prompts_individual = generated_response.strip().split('\n')  # Divide el Texto en Líneas Individuales
    # Procesar los prompts eliminando números y puntos al principio con expresion regular
    # prompts_regular = [re.sub(r'^\d+\.\s*', '', prompt) for prompt in prompts_individual]  # Usar una lista de strings como argumento de re.sub

    # Lista de prompts
    # image_prompts = [prompt.strip() for prompt in prompts_regular if prompt.strip() != ""] #compresion de lista
    image_prompts = [prompt.strip() for prompt in prompts_individual if prompt.strip() != ""]
    print("Prompts procesados:", image_prompts)  # Ver los prompts después de quitar líneas vacías

    # Ahora, llamamos a la api leonardo para generar imágenes

    return generate_images(request, image_prompts)  # Devolver directamente el resultado de la función


# API Leonardo
"""Solicita a Leonardo AI la generación de imágenes y devuelve el ID de la generación."""


# Modelo Leonardo Diffusion: "b820ea11-02bf-4652-97ae-9ac0cc00593d"
# Modelo Absolute Reality v1.6: "e316348f-7773-490e-adcd-46757c738eb7"


def generate_images(request, image_prompts):
    # Inicializar una lista vacia para almacenar todas las rutas de las imágenes
    all_image_paths = []

    # Para cada prompt, generar una imagen con Leonardo
    # se obtiene tanto el índice de prompt de la lista image_prompts como el valor del prompt
    for index, prompt in enumerate(image_prompts):
        # contiene las imágenes relacionadas con un prompt particular
        individual_image_paths = []
        retries = 3  # Número de reintentos permitidos
        while retries > 0:
            try:
                # Llamar a la API de Leonardo para solicitar 4 imágenes por prompt
                # Configuración de la clave API de Leonardo
                leonardo_api_key = LEONARDO_API_KEY
                url = "https://cloud.leonardo.ai/api/rest/v1/generations"
                payload = {
                    "height": 1024,
                    "modelId": "e316348f-7773-490e-adcd-46757c738eb7",
                    "prompt": prompt,
                    "negative_prompt": "blurry, blurry eyes, disfigured eyes, deformed eyes, abstract, disfigured, deformed,"
                                       " cartoon, animated, toy, figure, framed, 3d, out of frame, hands, cartoon, 3d, "
                                       "disfigured, bad art, deformed, deformed feet, feet misshaped, poorly drawn, "
                                       "extra limbs, close up, b&w, weird colors, blurry, watermark duplicate, morbid, "
                                       "mutilated, out of frame, extra feet, mutated feet, poorly drawn feet,"
                                       " poorly drawn toes, mutation, deformed, ugly, blurry, bad anatomy, bad proportions, "
                                       "extra limbs, cloned feet, disfigured, out of frame, ugly, extra limbs, bad anatomy, "
                                       "gross proportions, malformed limbs, missing arms, missing legs, extra arms, "
                                       "extra legs, mutated hands, fused fingers, too many toes, blurry, bad anatomy, "
                                       "extra limbs, poorly drawn face, poorly drawn toes, missing toes, mutated feet, "
                                       "fused toes, too many toes, long neck, blurry, bad anatomy, extra limbs, "
                                       "cloned face, poorly drawn face, poorly drawn feet, missing toes, ugly toes.",
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
                    static_dir = os.path.join(base_dir, 'video_tiktok', 'static', 'video_tiktok', 'img',
                                              'generated_images')
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
                break  # Salir del bucle de reintento si no hay errores

            # except Exception as e:
            # Imprimir la excepción
            #    print(f"Error al generar imagen para el prompt: {prompt}")
            #    print(e)
            # Devolver un mensaje de error
            #    return JsonResponse({"error": str(e)})

            except requests.RequestException as e:
                print(f"Error al generar imagen para el prompt {prompt}: {str(e)}")
                retries -= 1  # Decrementar el número de reintentos
                if retries == 0:
                    # Si se acabaron los reintentos, continuar con el siguiente prompt
                    break
                time.sleep(10)  # Esperar antes de intentar nuevamente

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


# Convertir texto a voz
def texto_a_voz(request):
    try:
        texto = request.POST.get('texto')
        selected_voice = request.POST.get('voice')
        audio_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'audios',
                                  'generated_audios', 'audio.mp3')
        if not os.path.exists(audio_path):
            # Si el archivo de audio no existe, generar el audio
            API_KEY = PLAY_API_KEY
            HEADERS = {
                'Authorization': 'Bearer ' + API_KEY,
                'Content-Type': 'application/json',
                "accept": "application/json",
                "X-User-ID": PLAY_USER
            }
            url = "https://play.ht/api/v1/convert"

            payload = {
                "content": [texto],
                "voice": selected_voice
            }

            response = requests.post(url, json=payload, headers=HEADERS)
            data = response.json()  # Convertir la respuesta en un objeto JSON
            transcriptionId = data.get('transcriptionId')

            print(data)

            if transcriptionId:
                voz_success = generate_voz(transcriptionId)
                if voz_success:
                    # Realizar la transcripción
                    transcribe_success = transcribe_audio(audio_path)
                    if transcribe_success:
                        # Si la transcripción tiene éxito, realizar la corrección de subtítulos
                        correccion_subtitulo(texto)

                return JsonResponse({"status": "success", "transcriptionId": transcriptionId})
            else:
                return JsonResponse({"status": "error", "message": "No se pudo obtener transcriptionId"}, status=400)

        # Verificar si el archivo se ha generado correctamente o ya existía
        if os.path.exists(audio_path):
            # Si existe el archivo de audio, intentar transcribirlo
            transcribe_success = transcribe_audio(audio_path)
            if transcribe_success:
                # Si la transcripción tiene éxito, realizar la corrección de subtítulos
                correccion_subtitulo(texto)
                return JsonResponse({"status": "success", "audioPath": audio_path})
            else:
                return JsonResponse({"status": "error", "message": "Error en la transcripción del audio"},
                                    status=400)
        else:
            return JsonResponse({"status": "error", "message": "No se pudo generar el audio"}, status=400)

    # maneja errores específicos de solicitud (por ejemplo, problemas de conexión o errores al llamar a la API externa
    except requests.exceptions.RequestException as e:
        print(f"Error en la solicitud a la API de texto a voz: {e}")
        return JsonResponse({"status": "error", "message": "Error en la solicitud texto a voz"}, status=500)
    # captura cualquier otro error inesperado que pueda surgir durante el proceso
    except Exception as ex:
        print(f"Error inesperado en texto_a_voz: {ex}")
        return JsonResponse({"status": "error", "message": "Error inesperado en texto_a_voz"}, status=500)


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
            time.sleep(5)  # Esperar 10 segundos antes del próximo intento

    print("No se pudo obtener la URL del audio después de varios intentos.")
    return JsonResponse(
        {"status": "error", "message": "No se pudo obtener la URL del audio después de varios intentos."}, status=400)


def transcribe_audio(audio_path):
    warnings.simplefilter(action='ignore', category=UserWarning)
    srt_directory = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                                 'generated_subtitulos', 'whisper')

    # Transcribir el audio del video por comando
    command = f"whisper {audio_path} --task transcribe --model small --output_format srt --output_dir {srt_directory} --language es"

    # Ejecuta el comando
    response = os.system(command)

    # Verificar si el comando se ejecutó con éxito
    if response == 0:
        print("Transcripción completada con éxito!")
        audioSrt = os.path.join(srt_directory, "audio.srt")

        # Cambiar el nombre del archivo transcribido
        transcribed_srt_path = os.path.join(srt_directory, "transcribed_audio.srt")
        os.rename(audioSrt, transcribed_srt_path)

        return transcribed_srt_path, True
        # return JsonResponse({'status': 'success', 'audioSrt': audioSrt})
    else:
        print("Hubo un error en la transcripción.")
        return False
        # return JsonResponse({'status': 'error'})


def correccion_subtitulo(texto):
    try:
        # Clave API
        openai.api_key = OPENAI_API_KEY

        srt_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                                'generated_subtitulos', 'whisper', 'transcribed_audio.srt')

        # Crear un nuevo archivo de subtítulos corregidos
        correction_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                                       'generated_subtitulos', 'original', "audio.srt")

        # Lee el contenido del archivo SRT
        with open(srt_path, 'r', encoding='utf-8') as archivo:
            contenido_srt = archivo.read()

        max_retries = 3  # Número máximo de intentos
        retry_count = 0

        while retry_count < max_retries:
            try:
                resultado = openai.Completion.create(
                    engine="text-davinci-003",
                    # model="gpt-3.5-turbo-instruct",
                    prompt=f'es: En base a este texto: {texto}.\n\nMe corriges la ortografia en el siguiente subtitulo, respetando la cantidad y marcas de tiempo?\n\n{contenido_srt}\n',
                    # max_tokens=2097 - len(tema),
                    max_tokens=2049,
                    # temperature=0.5
                )
                break  # Salir del bucle si no hay errores
            except openai.error.OpenAIError as e:
                print(f"Error de OpenAI: {e}")
                return JsonResponse({'error': 'Error al generar el subtitulo'}, status=500)
            except requests.exceptions.ConnectionError:
                retry_count += 1
                print(f"Error de conexión. Reintentando {retry_count}/{max_retries}...")
    except Exception as e:
        print(f"Error: {e}")
        return JsonResponse({'error': 'Error al leer el archivo SRT'}, status=500)
    else:
        generated_response = resultado.choices[0].text.strip()
        print(generated_response)

        with open(correction_path, 'w', encoding='utf-8') as corrected_file:
            corrected_file.write(generated_response)

        return JsonResponse({'generated_script': generated_response})
    # finally:
    #    archivo.close()


'''
def correccion_subtitulo2(texto):
    srt_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                            'generated_subtitulos', 'whisper', 'transcribed_audio.srt')

    # Usar expresión regular para dividir el texto en oraciones
    str_list = re.split(r'(?<!\d\.\d)\.\s', texto)

    # Agregar un punto al final de cada frase en str_list si no termina con uno
    for i in range(len(str_list)):
        if not str_list[i].endswith('.'):
            str_list[i] += '.'

    print(str_list)
    try:

        # Cargar los subtítulos del archivo srt existente
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()

        # Convertir los subtítulos a una lista de objetos Subtitle
        subtitles = list(srt.parse(srt_content))

        # Asignar tiempos a los subtítulos del texto
        for i, subtitle in enumerate(subtitles):
            if i < len(str_list):
                subtitle.content = str_list[i]  # Reemplazar el contenido del subtítulo con el texto correspondiente

        # Crear un nuevo archivo de subtítulos corregidos
        srt_directory = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                                     'generated_subtitulos', 'original', "audio.srt")

        # Componer y escribir los subtítulos corregidos en un nuevo archivo srt
        corrected_srt_content = srt.compose(subtitles)
        with open(srt_directory, "w", encoding="utf-8") as f:
            f.write(corrected_srt_content)
            print(corrected_srt_content)
        return "Corrección exitosa: Subtítulos actualizados correctamente."

    except FileNotFoundError:
        print(f"El archivo de subtítulos '{srt_path}' no se encontró en la ubicación especificada.")
        return "Error: Archivo de subtítulos no encontrado."
    except Exception as e:
        print(f"Se produjo un error: {e}")
        return "Error inesperado al corregir subtítulos."
'''


# Generacion de video

def generate_video(request):
    # Obtiene la música seleccionada desde el frontend
    selected_music_filename = request.POST['music']
    print("Música recibida:", request.POST['music'])

    music_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos', 'music',
                              selected_music_filename + '.mp3')

    try:
        image_dir = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'img', 'generated_images')

        # Filtrar solo las imágenes que contienen "_thumbnail" en su nombre
        image_filenames = sorted(
            [filename for filename in os.listdir(image_dir) if filename.endswith('.png') and "_thumbnail" in filename])

        # Agrupar las imágenes en conjuntos de 4
        # grouped_images = [image_filenames[i:i + 2] for i in range(0, len(image_filenames), 2)]
        grouped_images = [image_filenames[i:i + 4] for i in range(0, len(image_filenames), 4)]

        # Luego, selecciona una imagen aleatoriamente de cada grupo
        selected_images = [random.choice(group) for group in grouped_images]

        # Crear una lista de funciones y argumentos
        effects = [
            # Aplica una transición de deslizamiento desde la derecha a la imagen durante un segundo
            (transfx.slide_in, 1, 'right'),
            # Aplica una transición de deslizamiento hacia la izquierda a la imagen durante un segundo.
            (transfx.slide_out, 1, 'left'),
            # Aplica una transición de deslizamiento desde abajo a la imagen durante un segundo
            (transfx.slide_out, 1, 'bottom'),
            #  Aplica una transición de deslizamiento desde arriba a la imagen durante un segundo.
            (transfx.slide_in, 1, 'top'),
            # Aplica el efecto de crossfadein al clip con una duración de 1 segundo
            (transitions.crossfadein, 1),
            # Aplica el efecto de crossfadeout al clip con una duración de 1 segundo
            (transitions.crossfadeout, 1),
            # Aplica el efecto de make_loopable al clip con una duración de 1 segundo
            # (transitions.make_loopable, 3),
            # Aplica una transición de desvanecimiento (aparece gradualmente) a la imagen durante un segundo.
            (transitions.fadein, 1),
            # desaparece gradualmente
            (transitions.fadeout, 1),
            # Invierte los colores de la imagen.
            # (vfx.invert_colors,),
            # Aplica un efecto de pintura a la imagen durante un segundo.
            # (vfx.painting, 1)
            # Zoom
            # (paddedzoom,),
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

            # Llamar a paddedzoom con la ruta de la imagen actual
            # video_with_zoom = paddedzoom(img_path, image_duration, zoomfactor=2.0)

            # Si es el último clip, agrega un retraso (p.ej., 3 segundos) a su duración.
            if idx == len(selected_images) - 1:
                clip_duration = image_duration + 3
            else:
                clip_duration = image_duration

            clip = ImageClip(img_path, duration=clip_duration)

            # Aplica un efecto aleatorio de la lista de efectos a la imagen
            transition_func, *transition_args = random.choice(effects)
            try:
                clip = clip.fx(transition_func, *transition_args)
                print("Efecto", transition_func.__name__, "aplicado correctamente")
            except Exception as e:
                print("Error al aplicar el efecto", transition_func.__name__, ":", e)
            # Redimensiona la imagen a un 120% de su tamaño original
            # try:
            #    #clip = clip.fx(vfx.resize, newsize=[dim * 1.2 for dim in clip.size])
            #    clip = clip.fx(vfx.resize.resize, newsize=[dim * 1.2 for dim in clip.size])
            #    print("Efecto de resize aplicado correctamente")
            # except Exception as e:
            #    print("Error al aplicar el efecto de resize:", e)
            # Crea un clip compuesto por la imagen con su efecto de transición
            clip = CompositeVideoClip([clip])

            # Asigna el momento de inicio del clip compuesto
            clip = clip.set_start((image_duration - 2) * idx)

            clips.append(clip)

        # Concatenar y Exportar
        final_clip = concatenate_videoclips(clips, method="compose")
        video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                  'generated_videos',
                                  'video_final.mp4')
        final_clip.write_videofile(video_path, codec="libx264", fps=24)

        # Ruta del video con voz
        output_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                   'generated_videos', 'video_audio.mp4')

        # ruta  donde se guardará el video con música
        bajar_music = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'sonidos',
                                   'bajar_music')

        # Primero, integra el audio de voz en off con el video
        resultado_audio_video = audio_video(video_path, audio_path, output_path)
        print("Resultado de audio_video:", resultado_audio_video)
        if resultado_audio_video:
            print("Audio integrado con éxito en el video.")

            # Después, modifica la música con la función add_music
            ruta_musica_modificada, exito_musica = add_music(music_path, bajar_music)
            if exito_musica:
                print("Música modificada exitosamente.")

                # Combina el video con la música modificada
                output_folder = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                             'final_videos')
                print("Preparándose para llamar a combiner_video con música:", ruta_musica_modificada, "y video:",
                      resultado_audio_video)

                output_path_combinado = combiner_video(ruta_musica_modificada, resultado_audio_video, output_folder)

                if output_path_combinado:
                    print("Video combinado con éxito.")
                    output_path = output_path_combinado  # Actualiza la ruta del video final
                else:
                    print("Error al combinar el video y la música.")

            else:
                print("Error al modificar la música.")
        else:
            print("Hubo un error al integrar el audio en el video.")

        return JsonResponse({'videoPath': output_path})

    except Exception as e:
        return JsonResponse({"error": str(e)})


# Zoom 20%
def paddedzoom(img_path, image_duration, zoomfactor=0.8):
    '''
    Zoom in/out an image while keeping the input image shape.
    i.e., zero pad when factor<1, clip out when factor>1.
    there is another version below (paddedzoom2)
    '''
    img = Image.open(img_path)
    img_array = np.array(img)

    out = np.zeros_like(img_array)
    zoomed = cv2.resize(img_array, None, fx=zoomfactor, fy=zoomfactor)

    h, w, _ = img_array.shape
    zh, zw, _ = zoomed.shape

    if zoomfactor < 1:  # zero padded
        out[(h - zh) // 2:-(h - zh) // 2, (w - zw) // 2:-(w - zw) // 2] = zoomed
    else:  # clip out
        out = zoomed[(zh - h) // 2:-(zh - h) // 2, (zw - w) // 2:-(zw - w) // 2]

    # Convertir la imagen con zoom de nuevo a un objeto PIL
    img_with_zoom = Image.fromarray(out)

    # Ruta donde se guardarán las imágenes
    output_dir = os.path.join(settings.BASE_DIR, 'ZoomVideoComposer', 'example')

    # Guardar la imagen con zoom en la carpeta de salida
    img_with_zoom.save(os.path.join(output_dir, '00001.png'))

    # Guardar la imagen original con un nuevo nombre en la misma carpeta
    img.save(os.path.join(output_dir, '00002.png'))

    # Llamar al comando personalizado
    # call_command('zoom_video', folder='example', output='example_output.mp4', duration=20, direction='outin',
    #             easing='easeInOutSine')

    call_command('zoom_video', img=img, image_duration=image_duration, img_with_zoom=img_with_zoom)

    # Devolver las rutas de las imágenes
    # return zoom_output


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


def add_music(music_path, output_folder):
    try:
        # Extraer el nombre base del archivo de música
        base_name = os.path.basename(music_path)
        # Crear un nuevo nombre para el archivo de salida
        output_name = os.path.splitext(base_name)[0] + "_modified.mp3"

        # Ruta completa para el archivo de salida en la misma carpeta que el archivo original
        output_path = os.path.join(output_folder, output_name)

        audio = AudioSegment.from_mp3(music_path)
        volumen_musica = audio.dBFS
        print(f"Volumen de la música de fondo: {volumen_musica}")

        volumen_nuevo = volumen_musica - 20
        audio_modificado = audio.apply_gain(volumen_nuevo - volumen_musica)
        print(f"Volumen de la música ajustada: {audio_modificado.dBFS}")

        # Guardar el archivo de audio modificado
        audio_modificado.export(output_path, format="mp3")
        print("Se ajustó la música con éxito!")

        # Verificar si el archivo existe
        if os.path.exists(output_path):
            return output_path, True
        else:
            return None, False

    except Exception as e:
        print(f"Error al modificar la música: {e}")
        return None, False


def audio_video(video_path, audio_path, output_path):
    try:

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
        video_with_audio_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                                             'generated_videos', 'video_audio.mp4')
        video_with_audio.write_videofile(output_path, codec='libx264')
        print("Audio integrado con éxito!")
        return output_path  # Retorna True al finalizar con éxito

    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar un archivo: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

    return None  # Retorna None si ocurre una excepción


def combiner_video(original_music_path, video_path, output_folder):
    try:
        # Cargar el video con la voz añadida
        print("Cargando video con voz...")
        video_con_voz = mpe.VideoFileClip(video_path)

        # Cargar la música modificada
        print("Cargando música modificada...")
        musica_modificada = mpe.AudioFileClip(original_music_path)

        # Si la musica de fondo es mas largo que el video, corta la musica.
        if musica_modificada.duration > video_con_voz.duration:
            musica_modificada = musica_modificada.subclip(0, video_con_voz.duration)
        # Si el audio es más corto que el video, repite el audio
        elif musica_modificada.duration < video_con_voz.duration:
            repeticiones = int(video_con_voz.duration // musica_modificada.duration) + 1
            musica_modificada = mpe.concatenate_audioclips([musica_modificada] * repeticiones)
            musica_modificada = musica_modificada.subclip(0, video_con_voz.duration)

        # Aplicar el efecto de audio_fadeout al audio de voz:
        # para que baja el volumen de la musica 5 seg antes que termine
        musica_modificada = musica_modificada.audio_fadeout(5)

        # Combinar el video con voz y la música modificada
        print("Combinando audio y video...")
        audio_composite = mpe.CompositeAudioClip([video_con_voz.audio, musica_modificada])
        # Asignar el audio compuesto al vídeo
        video_combinado = video_con_voz.set_audio(audio_composite)

        print("Escribiendo archivo de video combinado...")
        output_path = os.path.join(output_folder, "video_combinado.mp4")
        video_combinado.write_videofile(output_path)
        print("Música integrada con éxito!")
        return output_path

    except FileNotFoundError as e:
        print(f"Error: No se pudo encontrar un archivo: {e}")
    except Exception as e:
        print(f"Error inesperado: {e}")

    return None  # Retorna None si ocurre una excepción


########################## SUBTITULOS ##########################


def procesar_subtitulos(request):
    # Definir la ruta al archivo SRT dentro de la función
    audio_srt = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                             'generated_subtitulos', 'original', 'audio.srt')
    # Ruta al archivo SRT ajustado
    adjusted_srt = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                                'generated_subtitulos', 'adjusted', 'audio_adjusted.srt')

    try:
        # Divide y ajusta los subtítulos
        divide_subtitle(audio_srt, adjusted_srt)

        output_path = aplicar_estilos_y_convertir(adjusted_srt)
        return JsonResponse({'message': 'Procesado con éxito', 'outputPath': output_path})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


def get_cps(sub):
    """ Calcula el promedio de caracteres por segundo (cps) de un subtítulo en particular.
        Utiliza el tiempo de inicio y el tiempo de finalización del subtítulo para determinar su duración
        y luego divide el número de caracteres en el texto del subtítulo por esa duración."""
    try:
        duration = (sub.end - sub.start).seconds
        if duration == 0:  # Evita la división por cero
            return 0
        return len(sub.text) / duration
    except Exception as e:
        print(f"Se produjo un error al calcular el cps: {e}")


def adjust_times(sub, split_texts, cps):
    try:
        start_time = sub.start.ordinal
        durations = [(len(text) / cps) for text in split_texts]

        times = []
        for duration in durations:
            end_ordinal = start_time + int(duration * 1000)  # Convertir a milisegundos
            end_time = pysrt.srttime.SubRipTime(0, 0, 0, end_ordinal)
            start_time_obj = pysrt.srttime.SubRipTime(0, 0, 0, start_time)

            times.append((start_time_obj, end_time))
            start_time = end_ordinal

        return times
    except Exception as e:
        print(f"Se produjo un error al ajustar los tiempos: {e}")


def divide_text(text, max_length=36):
    """Divide el texto en bloques más pequeños de aproximadamente max_length."""
    try:
        words = text.split()
        segments = []
        segment = ""

        for word in words:
            if len(segment) + len(word) > max_length and segment:
                segments.append(segment.strip())
                segment = ""
            segment += word + " "

        if segment:
            segments.append(segment.strip())

        return segments
    except Exception as e:
        print(f"Se produjo un error al dividir el texto: {e}")


def divide_subtitle(audio_srt, adjusted_srt, max_length=36):
    """
          Subdivide un subtítulo si excede una longitud máxima.
          :param subtitle: El subtítulo (objeto srt.Subtitle) a subdividir.
          :param max_length: Longitud máxima de palabras permitida por subtítulo.
          :return: Lista de subtítulos subdivididos.
          """
    try:
        # Llama a la función para dividir los subtítulos
        subs = pysrt.open(audio_srt, encoding='utf-8')

        # Ajustar todos los subtítulos en 2 segundos
        for sub in subs:
            sub.start = sub.start - SubRipTime(0, 0, 0.25)
            sub.end = sub.end - SubRipTime(0, 0, 0.25)

        for idx, sub in enumerate(subs, 1):
            cps = get_cps(sub)
            # print(f"Subtítulo {idx} tiene un cps de: {cps:.2f}")

        new_subs = pysrt.SubRipFile()

        for sub in subs:
            split_texts = divide_text(sub.text, max_length)
            cps = get_cps(sub)
            times = adjust_times(sub, split_texts, cps)

            for (start, end), split_text in zip(times, split_texts):
                new_sub = pysrt.SubRipItem()
                new_sub.start = start
                new_sub.end = end
                new_sub.text = split_text
                new_subs.append(new_sub)
        # Impresion de los subtitulos
        for entry in new_subs:
            print(entry)

        index = 1
        for entry in new_subs:
            entry.index = index
            index += 1

        new_subs.save(adjusted_srt, encoding='utf-8')
        # subs.save(adjusted_srt)
    except Exception as e:
        print(f"Se produjo un error al dividir el subtítulo: {e}")


def srt_to_ass(srt_path, ass_path):
    """Convierte un archivo .srt a .ass"""
    subs = pysubs2.load(srt_path)
    subs.save(ass_path)


def aplicar_estilos_y_convertir(adjusted_srt):
    # Utiliza el parámetro audio_srt para definir la ruta al archivo SRT
    srt_path = adjusted_srt

    # Ruta del archivo ASS de salida
    ass_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'subtitulos',
                            'generated_subtitulos', 'modificado', 'subtitulo.ass')

    video_path = os.path.join(settings.BASE_DIR, 'video_tiktok', 'static', 'video_tiktok', 'videos',
                              'final_videos', 'video_combinado.mp4')

    # Llamamos a la funcion para convertir SRT a ASS
    srt_to_ass(srt_path, ass_path)

    # Aplicar estilos al archivo ASS
    subs = pysubs2.load(ass_path)
    default_style = subs.styles.get("Default", pysubs2.SSAStyle())

    # Modificar el estilo
    default_style.fontname = "Fiest Slant"
    default_style.fontsize = 10
    default_style.marginv = 10
    default_style.primarycolor = '&H00FFFFFF'  # amarillo '&H00DBFC&'# negro '&H000000&'
    default_style.outlinecolor = '&H000000&'
    default_style.outline = 1  # Grosor del contorno aumentado
    default_style.shadow = 0
    default_style.alignment = 5

    # default_style.bold = True

    subs.styles["Default"] = default_style
    subs.save(ass_path)
    print(f"Archivo ASS guardado en: {ass_path}")

    # Ruta del video de salida
    output_path = os.path.join(os.path.dirname(video_path), 'video_sincronizado.mp4')

    # Rutas relativas
    video_rel = "video_tiktok/static/video_tiktok/videos/final_videos/video_combinado.mp4"
    output_rel = "ffmpeg/video_subtitulos.mp4"
    ass_rel = "video_tiktok/static/video_tiktok/subtitulos/generated_subtitulos/modificado/subtitulo.ass"

    # Integrar los subtítulos al video

    os.system(f"ffmpeg -i {video_rel} -vf ass={ass_rel} {output_rel}")

    print("Proceso ffmpeg finalizado")

    return output_path


# os.system(f"ffmpeg -i {video_rel} -vf ass={ass_rel} {output_rel}")


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

