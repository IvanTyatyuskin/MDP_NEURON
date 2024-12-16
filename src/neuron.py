import face_recognition
import os
import numpy as np
import cv2
from collections import defaultdict
from ultralytics import YOLO
import requests
from api import visitMark, visitCreate

# Путь к датасету
dataset_path = 'src/faces-dataset'

# Функция для загрузки изображений и кодирования лиц
def load_known_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = face_recognition.load_image_file(image_path)
                face_encoding = face_recognition.face_encodings(image)
                if face_encoding:
                    known_face_encodings.append(face_encoding[0])
                    known_face_names.append(person_name)

    return known_face_encodings, known_face_names

def startNeuralNetwork(pathToVideo: str, eventID: str):
    known_face_encodings, known_face_names = load_known_faces(dataset_path)

    # Инициализация модели YOLO
    model = YOLO("src/models/yolov11s50epochs.pt")

    # Путь к видео
    video_path = pathToVideo
    cap = cv2.VideoCapture(video_path)

    # Проверка успешного открытия видео
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Параметры для сохранения видео
    output_path = 'src/videos/last_detected_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Хранение истории треков и распознанных лиц
    track_history = defaultdict(lambda: [])
    recognized_faces = {}
    face_encodings_history = {}
    recognized_names = []
    created_visits = []

    # Цикл обработки кадров видео
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Запуск YOLO для трекинга
            results = model.track(frame, classes=1, conf=0.4, persist=True) #class 0 - человек целиком, class 1 - только голова

            # Проверка наличия обнаруженных объектов
            if results[0].boxes is not None and results[0].boxes.id is not None:
                # Получение bounding boxes и ID треков
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Визуализация результатов на кадре
                annotated_frame = results[0].plot()

                # Обработка каждого bounding box
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]

                    # Увеличение размера bounding box
                    top, left = int(y - h / 2) - 10, int(x - w / 2) - 10
                    bottom, right = int(y + h / 2) + 10, int(x + w / 2) + 10

                    # Проверка границ bounding box
                    if top < 0: top = 0
                    if left < 0: left = 0
                    if bottom > frame_height: bottom = frame_height
                    if right > frame_width: right = frame_width

                    # Извлечение лица из кадра
                    face_image = frame[top:bottom, left:right]

                    # Проверка, что изображение лица не пустое
                    if face_image.size == 0:
                        continue

                    face_locations = face_recognition.face_locations(face_image)

                    # Проверка наличия лица в bounding box
                    if not face_locations:
                        continue

                    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

                    try:
                        face_encodings = face_recognition.face_encodings(face_rgb, face_locations)
                        cv2.imshow(face_rgb)
                        #сохранить файл в faces
                    except Exception as e:
                        print(f"Error processing face encoding for track_id {track_id}: {e}")
                        continue

                    # Распознавание лица
                    if track_id not in recognized_faces:
                        recognized_faces[track_id] = "Unknown"

                    if face_encodings:
                        face_encoding = face_encodings[0]
                        face_encodings_history[track_id] = face_encoding

                        if recognized_faces[track_id] == "Unknown":
                            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.46)
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                recognized_faces[track_id] = known_face_names[best_match_index]
                                if (known_face_names[best_match_index] not in recognized_names):
                                    print("Create a folder for new recognized person")
                                    visitID = visitCreate(known_face_names[best_match_index], eventID)
                                    created_visits.append({track_id, visitID})
                                    recognized_names.append(known_face_names[best_match_index])
                                print("Send data of recognized person to folder")
                                cv2.imwrite(f"src/faces/{track_id}{known_face_names[best_match_index]}.jpg", face_rgb)
                                for elem in created_visits:
                                    if elem[0] == track_id:
                                        temp_visit_id = elem[1]
                                visitMark(temp_visit_id, f"src/faces/{track_id}{known_face_names[best_match_index]}.jpg")
                            else:
                                # Проверка наличия лица в истории
                                for hist_track_id, hist_encoding in face_encodings_history.items():
                                    if face_recognition.compare_faces([hist_encoding], face_encoding)[0]:
                                        recognized_faces[track_id] = recognized_faces[hist_track_id]
                                        print("Send data of unknown person to folder")
                                        cv2.imwrite(f"src/faces/{track_id}unknown.jpg", face_rgb)
                                        for elem in created_visits:
                                            if elem[0] == track_id:
                                                temp_visit_id = elem[1]
                                        visitMark(temp_visit_id, f"src/faces/{track_id}unknown.jpg")
                                        break
                                    else:
                                        print("Create a folder for new unknown person")
                                        visitID = visitCreate(None, eventID)
                                        print("Send data of unknonw person to folder")
                                        cv2.imwrite(f"src/faces/{track_id}unknown.jpg", face_rgb)
                                        visitMark(visitID, f"src/faces/{track_id}unknown.jpg")
                                        break

                    # Отображение метки распознанного лица
                    name = recognized_faces[track_id]
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(annotated_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(annotated_frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                # Запись кадра в выходное видео
                out.write(annotated_frame)
            else:
                # Если объекты не обнаружены, записываем исходный кадр
                out.write(frame)
        else:
            break

    # Освобождение ресурсов
    cap.release()
    out.release()