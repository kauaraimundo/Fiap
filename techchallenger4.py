import cv2
import os
import numpy as np
from tqdm import tqdm
from deepface import DeepFace
import mediapipe as mp
import math

# Variáveis para contagem
anomalous_movement_counter = 0
analyzed_frames_counter = 0

# Função que processa o vídeo 
def proccess_video(video_path, output_path):
    global analyzed_frames_counter
    global anomalous_movement_counter
    
    # Capturar vídeo do arquivo especificado
    cap = cv2.VideoCapture(video_path)
    
    # Verificar se o vídeo foi aberto corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter propriedades do vídeo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    batch_size = 64 # processamento em batch para melhorar a performance
    
    batch = []
    countBatch = 0

    previous_landmarks_batch = None #landmarks anterior para detecção de movimentos anomalos 
    # Loop para processar cada frame do vídeo com barra de progresso
    for i in tqdm(range(total_frames), desc="Processando vídeo"):
        
        # Ler um frame do vídeo
        ret, frame = cap.read()
       
        # Se não conseguiu ler o frame (final do vídeo), sair do loop
        if not ret:
            break

        batch.append(frame)

        if len(batch) >= batch_size:
            previous_landmarks_batch = process_batch(batch, out, countBatch, previous_landmarks_batch)
            countBatch = countBatch + 1
            batch = []  # Reinicia o batch após o processamento

    #processa os frames restantes        
    if batch:
        process_batch(batch, out, countBatch, previous_landmarks_batch)

    # Liberar a captura de vídeo e fechar todas as janelas
    cap.release()
    out.release()
    write_summary("Total de Movimentos Anômalos: " + str(anomalous_movement_counter))
    write_summary("Total de frames analisados: " + str(analyzed_frames_counter))
    write_summary("Total de frames." + str(total_frames))

#Função que processa os frames em pacotes para agilizar o processo 
#e retorna o último frame processado para analise de movimentos anomalos 
def process_batch(batch, out, countBatch, previous_landmarks_batch):
    global analyzed_frames_counter
    global anomalous_movement_counter

    previous_landmarks = previous_landmarks_batch #landmarks anterior para detecção de movimentos anomalos 

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(model_complexity=2,   # Nível mais alto de complexidade para maior precisão
                        static_image_mode=False, # False para utilizar o contexto de vídeo e melhorar o rastreamento
                        smooth_landmarks=True, # Suaviza os pontos para reduzir o ruído 
                        min_detection_confidence = 0.7,  # Confiança mínima para detectar a pose
                        min_tracking_confidence=0.5  # Confiança mínima para rastrear a pose
                        )

    mp_drawing = mp.solutions.drawing_utils 

    for i, frame in enumerate(batch):
        
        current_frame = i + (countBatch * 64) 
         # Escreve o número do frame no canto superior esquerdo
        text = f"Frame: {current_frame}"
        position = (10, 30)  # Posição do texto (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # Tipo de fonte
        font_scale = 1  # Tamanho do texto
        color = (0, 255, 0)  # Cor do texto (verde)
        thickness = 2  # Espessura do texto
        # Adiciona o texto ao frame
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

        if i % 4 == 0:  # analise de 4 em 4 frames  
            analyzed_frames_counter = analyzed_frames_counter + 1

            #Identificando as posições
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_frame)
            if results_pose.pose_landmarks :
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                current_landmarks = results_pose.pose_landmarks.landmark

                if is_arm_up(current_landmarks, mp_pose):
                   write_summary("Frame: " + str(current_frame) + " Pessoa com os braços levantado.")

                if is_person_lying(current_landmarks, mp_pose) : 
                    write_summary("Frame: " + str(current_frame) + " Pessoa deitada.")

                if(is_arms_down(current_landmarks, mp_pose)):
                    write_summary("Frame: " + str(current_frame) + " Pessoa com os braços abaixados.")

                if(is_looking_forward(current_landmarks, mp_pose)):
                    write_summary("Frame: " + str(current_frame) + " Pessoa olhando para frente") 

                look_side = is_profile_view(current_landmarks, mp_pose)
                if look_side is not None:
                    write_summary("Frame: " + str(current_frame) + " Pessoa olhando para o lado " + look_side)   

                if(previous_landmarks != None):
                    anomalous_movement = is_anomalous_movement(current_landmarks, previous_landmarks)
                    if(anomalous_movement):
                        write_summary("Frame: " + str(current_frame) + " Movimento anômalo identificado.")
                        anomalous_movement_counter = anomalous_movement_counter+1

                previous_landmarks = current_landmarks

            # Analisar o frame para detectar faces e expressões
            # detector_backend = mtcnn Melhorou o reconhecimento de faces de lado 
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')

            # Iterar sobre cada face detectada pelo DeepFace
            for face in result:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                
                # Obter a emoção dominante
                dominant_emotion = face['dominant_emotion']

                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Escrever a emoção dominante acima da face
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                if dominant_emotion:
                    write_summary("Frame: " + str(current_frame) + ' Emoção detectada - ' + dominant_emotion)

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    return previous_landmarks

# Função para verificar se o braço está levantado
def is_arm_up(landmarks, mp_pose):
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]

    left_arm_up = left_elbow.y < left_eye.y
    right_arm_up = right_elbow.y < right_eye.y

    return left_arm_up or right_arm_up

# Função para verificar se a pessoa está deitada
def is_person_lying(landmarks, mp_pose):
    # Coordenadas dos olhos
    left_eye_y = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value].y
    right_eye_y = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value].y
    # Coordenadas dos ombros
    left_shoulder_y = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
    right_shoulder_y = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y

    # Média das posições (alinhamento médio)
    eyes_y = (left_eye_y + right_eye_y) / 2
    shoulders_y = (left_shoulder_y + right_shoulder_y) / 2

    # Diferença entre olhos e ombros no eixo Y
    y_difference = abs(eyes_y - shoulders_y)
    
    # Diferença no eixo X entre os ombros (largura)
    left_shoulder_x = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x
    right_shoulder_x = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    shoulder_width = abs(left_shoulder_x - right_shoulder_x)

    # Tolerância para determinar se está deitado
    if y_difference < 0.1 and shoulder_width > 0.3:  # Ajuste os valores conforme necessário
        return True
    return False

# Função para verificar se a pessoa está com os braços abaixados
def is_arms_down(landmarks, mp_pose):
   
    # Get relevant landmarks
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    # Check if elbows and wrists are below shoulders (higher y value)
    arms_down = (
        left_elbow.y > left_shoulder.y and left_wrist.y > left_elbow.y and
        right_elbow.y > right_shoulder.y and right_wrist.y > right_elbow.y
    )

    return arms_down

# Função para verificar se a pessoa olhando para frente 
def is_looking_forward(landmarks, mp_pose):
    
    # Obtém os landmarks relevantes
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Critério 1: Nariz centralizado entre os ombros (eixo X)
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    nose_centered = abs(nose.x - shoulder_center_x) < 0.1  # Tolerância para centralização

    # Critério 2: Ombros alinhados (diferença mínima no eixo Y)
    shoulders_aligned = abs(left_shoulder.y - right_shoulder.y) < 0.05

    # Retorna True se ambos os critérios forem atendidos
    return nose_centered and shoulders_aligned

# Função para verificar se a pessoa de perfil
def is_profile_view(landmarks, mp_pose):

    # Obtém os landmarks relevantes
    nose = landmarks[mp_pose.PoseLandmark.NOSE]
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]

    # Critério 1: Nariz deslocado significativamente em direção a um ombro
    if nose.x < left_shoulder.x:  # Nariz mais próximo do ombro esquerdo
        return "Esquerdo"
    elif nose.x > right_shoulder.x:  # Nariz mais próximo do ombro direito
        return "Direto"

    return None

# Funçao para calcular a distância entre dois pontos 3D
def calculate_distance_3d(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2)

# Funçao para detectar movimentos Anomalos
def is_anomalous_movement(current_landmarks, previous_landmarks, threshold=0.18):
    if not previous_landmarks:
        return False

    # Calcula o movimento médio entre os pontos
    movement = sum(
        calculate_distance_3d(current, previous)
        for current, previous in zip(current_landmarks, previous_landmarks)
    ) / len(current_landmarks)

    return movement > threshold

# Função para escrever o resumo automático
def write_summary(text):
    with open("summary.txt", "a") as arquivo:
        arquivo.write(text + "\n")


with open("summary.txt", "w") as arquivo:
    pass

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output_video.mp4')  # Nome do vídeo de saída

# Chamar a função para detectar emoções e reconhecer faces no vídeo, salvando o vídeo processado
proccess_video(input_video_path, output_video_path)