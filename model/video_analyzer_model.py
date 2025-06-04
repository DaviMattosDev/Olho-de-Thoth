import cv2
import numpy as np
import mediapipe as mp
import subprocess
import json
import os
import tensorflow as tf
from tensorflow.keras.applications import (Xception, EfficientNetB0, InceptionV3, ResNet50)
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# --------------------------
# Função 1: Obter número total de frames do vídeo
# --------------------------
def get_video_frame_count(video_path):
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    cmd = [
        ffprobe_path,
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'default=nw=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        output = result.stdout.decode()
        for line in output.splitlines():
            if "nb_read_packets" in line:
                return int(line.split("=")[1])
    except Exception as e:
        print(f"[ERRO] Não foi possível ler quantidade de frames: {e}")
    return 30  # fallback

# --------------------------
# Função 2: Extrair frames
# --------------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

# --------------------------
# Função 3: Detectar faces e piscadelas
# --------------------------
mp_face_mesh = mp.solutions.face_mesh

def detect_faces_and_micro(frames, log_callback=None):
    face_issues = 0
    blink_count = 0
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for idx, frame in enumerate(frames):
            if log_callback:
                log_callback(f"[INFO] Analisando rosto - Frame {idx + 1}/{len(frames)}")
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            if results.multi_face_landmarks is None:
                face_issues += 1
            else:
                landmarks = results.multi_face_landmarks[0].landmark
                left_eye = [landmarks[i] for i in [386, 385, 380, 374, 368, 387]]
                right_eye = [landmarks[i] for i in [159, 158, 153, 145, 144, 160]]

                def eye_aspect_ratio(eye):
                    A = abs(eye[1].y - eye[5].y)
                    B = abs(eye[2].y - eye[4].y)
                    C = abs(eye[0].x - eye[3].x)
                    return (A + B) / (2.0 * C)

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                if ear < 0.2:
                    blink_count += 1
    return {"face_issues": face_issues, "blink_count": blink_count}

# --------------------------
# Função 4: Estimar nitidez
# --------------------------
def estimate_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm

# --------------------------
# Função 5: Jitter entre frames
# --------------------------
def detect_frame_jitter(frames, log_callback=None):
    diffs = []
    for i in range(1, len(frames)):
        if log_callback:
            log_callback(f"[INFO] Analisando tremor - Frame {i + 1}/{len(frames)}")
        diff = cv2.absdiff(frames[i], frames[i - 1])
        mean_diff = diff.mean()
        diffs.append(mean_diff)
    return diffs

# --------------------------
# Função 6: FFT (frequência espacial)
# --------------------------
def analyze_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    mean_magnitude = np.mean(magnitude_spectrum)
    return mean_magnitude

# --------------------------
# Função 7: Analisar metadados
# --------------------------
def analyze_metadata(video_path, log_callback=None):
    if log_callback:
        log_callback("[INFO] Lendo metadados do arquivo...")
    ffprobe_path = os.path.join("ffmpeg", "bin", "ffprobe.exe")
    if not os.path.isfile(ffprobe_path):
        raise FileNotFoundError(f"Arquivo ffprobe não encontrado em: {ffprobe_path}")
    cmd = [
        ffprobe_path,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_format',
        '-show_streams',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        metadata = json.loads(result.stdout)
    except json.JSONDecodeError:
        if log_callback:
            log_callback("[ERRO] Não foi possível ler os metadados do vídeo.")
        metadata = {}
    return metadata

# --------------------------
# Função 8: Carregar modelo treinado
# --------------------------
def load_model(model_name, log_callback=None):
    preprocess = None
    if model_name == "Xception":
        base_model = Xception(weights='imagenet', include_top=False)
        preprocess = xception_preprocess
    elif model_name == "EfficientNet":
        base_model = EfficientNetB0(weights='imagenet', include_top=False)
        preprocess = efficientnet_preprocess
    elif model_name == "Inception":
        base_model = InceptionV3(weights='imagenet', include_top=False)
        preprocess = inception_preprocess
    elif model_name == "ResNet":
        base_model = ResNet50(weights='imagenet', include_top=False)
        preprocess = resnet_preprocess
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")
    x = GlobalAveragePooling2D()(base_model.output)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model, preprocess

# Classificar frame com modelo escolhido
def classify_frame_with_ai(model, frame, preprocess_func):
    img = cv2.resize(frame, (299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_func(img_array)
    prediction = model.predict(img_array, verbose=0)
    fake_prob = prediction[0][1]
    return fake_prob

# --------------------------
# Função 9: Análise individual com um modelo
# --------------------------
def analyze_with_single_model(video_path, log_callback=None, model_type="Xception", total_frames=0, progress_callback=None):
    frames = extract_frames(video_path)
    model, preprocess = load_model(model_type, log_callback)
    ai_scores = []
    for i, frame in enumerate(frames):
        ai_score = classify_frame_with_ai(model, frame, preprocess)
        ai_scores.append(ai_score)
        if log_callback and i % 5 == 0:
            log_callback(f"[INFO] [{model_type}] Processando IA - Frame {i + 1}/{len(frames)}")
        if progress_callback and total_frames > 0:
            progress_callback(int((i + 1) / total_frames * 100))
    avg_ai_score = np.mean(ai_scores)
    return {
        "model": model_type,
        "avg_ai_score": avg_ai_score
    }

# --------------------------
# Função principal para análise (agora usa apenas o modelo selecionado)
# --------------------------
def analyze_video(video_path, log_callback=None, progress_callback=None, model_type="Xception"):
    frames = extract_frames(video_path)
    if log_callback:
        log_callback(f"[INFO] Extraídos {len(frames)} frames.")

    face_data = detect_faces_and_micro(frames, log_callback)
    blur_scores = [estimate_blurriness(frame) for frame in frames]
    jitter_scores = detect_frame_jitter(frames, log_callback)
    fft_scores = [analyze_fft(frame) for frame in frames]
    metadata = analyze_metadata(video_path, log_callback)

    avg_blur = np.mean(blur_scores)
    avg_jitter = np.mean(jitter_scores) if jitter_scores else 0
    avg_fft = np.mean(fft_scores)

    result = analyze_with_single_model(video_path, log_callback, model_type=model_type, total_frames=len(frames), progress_callback=progress_callback)
    avg_ai_score = result["avg_ai_score"]

    score = 0
    if face_data['face_issues'] > 5:
        score += 1
    if face_data['blink_count'] < 2:
        score += 0.5
    if avg_blur < 100:
        score += 0.5
    if avg_jitter > 10:
        score += 0.5
    if avg_fft > 200:
        score += 0.5
    if avg_ai_score > 0.6:
        score += 1.5

    meta_text = json.dumps(metadata).lower()
    ai_keywords = ["google", "lavf", "ai", "synthetic", "fake"]
    found_keywords = [kw for kw in ai_keywords if kw in meta_text]
    if found_keywords:
        score += 1

    return {
        "face_issues": face_data['face_issues'],
        "blink_count": face_data['blink_count'],
        "avg_blur": avg_blur,
        "avg_jitter": avg_jitter,
        "avg_fft": avg_fft,
        "avg_ai_score": avg_ai_score * 100,
        "score": score,
        "found_keywords": found_keywords,
        "deepfake": score >= 3.5,
        "metadata": metadata,
        "per_model": [result]
    }