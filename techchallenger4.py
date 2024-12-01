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
    
    moviment = ""
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

        if i % 10 == 0:  # analise de 10 em 10 frames 
            moviment =  '' 
            analyzed_frames_counter = analyzed_frames_counter + 1

            #Identificando as posições
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(rgb_frame)
           

            if results_pose.pose_landmarks :
                mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                current_landmarks = results_pose.pose_landmarks.landmark

                if is_arm_up(current_landmarks, mp_pose):
                   write_summary("Frame: " + str(current_frame) + " Pessoa com os bracos levantado.")
                   moviment = " Pessoa com os bracos levantado."

                if is_person_lying(current_landmarks, mp_pose) : 
                    write_summary("Frame: " + str(current_frame) + " Pessoa deitada.")
                    moviment += " Pessoa deitada."

                if(is_arms_down(current_landmarks, mp_pose)):
                    write_summary("Frame: " + str(current_frame) + " Pessoa com os bracos abaixados.")
                    moviment += " Pessoa com os bracos abaixados."

                if(is_looking_forward(current_landmarks, mp_pose)):
                    write_summary("Frame: " + str(current_frame) + " Pessoa olhando para frente") 
                    moviment += " Pessoa olhando para frente"
                else:
                    look_side = is_profile_view(current_landmarks, mp_pose)
                    if look_side is not None:
                        write_summary("Frame: " + str(current_frame) + " Pessoa olhando para o lado " + look_side)   
                        moviment += " Pessoa olhando para o lado " + look_side

                if(previous_landmarks != None):
                    anomalous_movement = is_anomalous_movement(current_landmarks, previous_landmarks)
                    if(anomalous_movement):
                        write_summary("Frame: " + str(current_frame) + " Movimento anomalo identificado.")
                        anomalous_movement_counter = anomalous_movement_counter+1
                        moviment += " Movimento anomalo identificado."

                previous_landmarks = current_landmarks

            # Analisar o frame para detectar faces e expressões
            # detector_backend = mtcnn Melhorou o reconhecimento de faces de lado 
            resultFaces = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, detector_backend='mtcnn')

            # Iterar sobre cada face detectada pelo DeepFace
            for face in resultFaces:
                # Obter a caixa delimitadora da face
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                
                # Obter a emoção dominante
                dominant_emotion = face['dominant_emotion']

                # Desenhar um retângulo ao redor da face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Escrever a emoção dominante acima da face
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
                if dominant_emotion:
                    write_summary("Frame: " + str(current_frame) + ' Emocao detectada - ' + dominant_emotion)

        # Escreve o movimento no frame
        text = f"Movimento: {moviment}"
        position = (10, 50)  # Posição do texto (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX  # Tipo de fonte
        font_scale = 0.7  # Tamanho do texto
        color = (0, 255, 0)  # Cor do texto (verde)
        thickness = 2  # Espessura do texto
        # Adiciona o texto ao frame
        cv2.putText(frame, text, position, font, font_scale, color, thickness)

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
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

    # Coordenadas dos ombros
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Coordenadas do nariz 
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Diferença em 'x' e 'y' entre os ombros
    shoulder_x_difference = abs(left_shoulder.x - right_shoulder.x)

    # Diferença em 'x' e 'y' entre os olhos
    eyes_x_difference = abs(left_eye.x - right_eye.x)
    
    # Critérios para estar deitado de lado:
    # 1. Ombros alinhados horizontalmente (pequena diferença em 'x')
    # 2. Olhos alinhados horizontalmente (pequena diferença em 'x')
    # 3. O nariz está entre os olhos no eixo 'y'
    if shoulder_x_difference < 0.1 and eyes_x_difference < 0.1:
        if(min(left_eye.y, right_eye.y) < nose.y < max(left_eye.y, right_eye.y)):
            return True  # Pessoa está deitada de lado

    return False  # Pessoa não está deitada de lado

# Função para verificar se a pessoa está com os braços abaixados
def is_arms_down(landmarks, mp_pose):
   
    # Coordenadas dos ombros, cotovelos e pulso
    left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
    right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
    right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]

    #Ombro acima do cotovelo e pulso abaixo do cotovelo
    arms_down = (
        left_elbow.y > left_shoulder.y and left_wrist.y > left_elbow.y and
        right_elbow.y > right_shoulder.y and right_wrist.y > right_elbow.y
    )

    return arms_down

# Função para verificar se a pessoa olhando para frente 
def is_looking_forward(landmarks, mp_pose):
    
     # Coordenadas do nariz e olhos
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
    
    # Verificar alinhamento horizontal do nariz entre os olhos
    nose_centered = min(left_eye.x, right_eye.x) < nose.x < max(left_eye.x, right_eye.x)
    
    # Verificar alinhamento vertical do nariz em relação aos olhos
    nose_y = nose.y
    eyes_y = (left_eye.y + right_eye.y) / 2
    nose_aligned_vertically = abs(nose_y - eyes_y) < 0.1  # Diferença pequena na altura dos olhos e nariz
    
    return nose_centered and nose_aligned_vertically

# Função para verificar se a pessoa de perfil
def is_profile_view(landmarks, mp_pose):

    # Coordenadas dos olhos e nariz
    left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

    # Diferença em 'x' entre os olhos
    eye_center_x = (left_eye.x + right_eye.x) / 2

    # Verifica a posição do nariz em relação ao centro dos olhos
    if nose.x < eye_center_x:  # Nariz está mais próximo do olho direito
        return "Esquerdo"  # Perfil Esquerdo
    elif nose.x > eye_center_x:  # Nariz está mais próximo do olho esquerdo
        return "Direito"  # Perfil Direito

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


# Apagando as informações anteriores no arquivo de resumo
with open("summary.txt", "w") as arquivo:
    pass

# Caminho para o arquivo de vídeo na mesma pasta do script
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'video.mp4')  # Substitua 'meu_video.mp4' pelo nome do seu vídeo
output_video_path = os.path.join(script_dir, 'output_video.mp4')  # Nome do vídeo de saída

# Chamar a função para detectar emoções e reconhecer posições no vídeo, salvando o vídeo processado
proccess_video(input_video_path, output_video_path)