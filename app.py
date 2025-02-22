from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLOv8 untuk deteksi barang mencurigakan
model = YOLO("yolov8n.pt")

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Variabel untuk tracking kepala
turn_count = 0
nod_count = 0
total_suspicious_actions = 0
total_possible_actions = 1200  # Untuk ujian 2 jam (120 menit)

# Bobot untuk setiap tindakan mencurigakan
WEIGHT_TURN = 1.0
WEIGHT_NOD = 1.5
WEIGHT_SUSPICIOUS_ITEM = 2.0
WEIGHT_CALCULATOR = 2.5

def generate_frames():
    global turn_count, nod_count, total_suspicious_actions
    cap = cv2.VideoCapture(1)  # Buka webcam

    # Variabel untuk kalibrasi
    calibration_done = False
    initial_nose_y = None

    # Inisialisasi variabel untuk tracking
    is_nodding = False
    is_turning = False
    nod_start_time = None
    turn_start_time = None
    nod_duration = 0
    turn_duration = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi ke RGB untuk MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Deteksi wajah
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Ambil landmark penting untuk tracking kepala
                left_eye = np.array([face_landmarks.landmark[33].x * w, face_landmarks.landmark[33].y * h])
                right_eye = np.array([face_landmarks.landmark[263].x * w, face_landmarks.landmark[263].y * h])
                nose_tip = np.array([face_landmarks.landmark[1].x * w, face_landmarks.landmark[1].y * h])
                chin = np.array([face_landmarks.landmark[152].x * w, face_landmarks.landmark[152].y * h])

                # Kalibrasi posisi awal kepala sebagai referensi normal
                if not calibration_done:
                    initial_nose_y = nose_tip[1]
                    calibration_done = True

                # Hitung bounding box kepala
                x_min = int(min(left_eye[0], right_eye[0]) - 50)
                x_max = int(max(left_eye[0], right_eye[0]) + 50)
                y_min = int(nose_tip[1] - 70)
                y_max = int(chin[1] + 20)

                # Hitung sudut kemiringan kepala (menoleh)
                eye_line = right_eye - left_eye
                angle = np.arctan2(eye_line[1], eye_line[0]) * 180 / np.pi

                # Hitung apakah menunduk (hanya jika sudah dikalibrasi)
                if calibration_done:
                    nod_threshold = 15  # Ambang batas perubahan vertikal
                    if nose_tip[1] - initial_nose_y > nod_threshold:
                        is_nodding = True
                        if nod_start_time is None:
                            nod_start_time = time.time()
                            nod_count += 1
                            total_suspicious_actions += WEIGHT_NOD  # Tambahkan bobot untuk menunduk
                    else:
                        if is_nodding:
                            nod_duration = time.time() - nod_start_time
                        is_nodding = False
                        nod_start_time = None

                # Deteksi menoleh ke kiri atau kanan
                if angle > 10:
                    status = "Ayo Fokus!!"
                    color = (0, 0, 255)  # Merah
                    if not is_turning:
                        is_turning = True
                        turn_start_time = time.time()
                        turn_count += 1
                        total_suspicious_actions += WEIGHT_TURN  # Tambahkan bobot untuk menoleh
                elif angle < -10:
                    status = "Ayo Fokus!!"
                    color = (0, 0, 255)  # Merah
                    if not is_turning:
                        is_turning = True
                        turn_start_time = time.time()
                        turn_count += 1
                        total_suspicious_actions += WEIGHT_TURN  # Tambahkan bobot untuk menoleh
                else:
                    status = "Semangat Ujian!!"
                    color = (0, 255, 0)  # Hijau
                    if is_turning:
                        turn_duration = time.time() - turn_start_time
                        is_turning = False

                if is_nodding:
                    status = "Liat Apa dibawah?"
                    color = (0, 0, 255)

                # Gambar kotak di area kepala
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

                # Tampilkan informasi di layar
                cv2.putText(frame, status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                cv2.putText(frame, f"Turn Count: {turn_count}", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f"Nod Count: {nod_count}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # Tampilkan durasi menoleh / nunduk
                if is_turning:
                    duration_text = f"Turn Time: {round(time.time() - turn_start_time, 2)}s"
                elif is_nodding:
                    duration_text = f"Nod Time: {round(time.time() - nod_start_time, 2)}s"
                else:
                    duration_text = f"Last Turn: {round(turn_duration, 2)}s | Last Nod: {round(nod_duration, 2)}s"

                cv2.putText(frame, duration_text, (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # === DETEKSI BARANG MENCURIGAKAN & KALKULATOR HP === #
        results = model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0].item()
                label = result.names[int(box.cls[0])]

                if label in ["knife", "scissors", "backpack"] and confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, f"{label.upper()} DETECTED!", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    total_suspicious_actions += WEIGHT_SUSPICIOUS_ITEM  # Tambahkan bobot untuk barang mencurigakan

                elif label in ["cell phone", "remote"] and confidence > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, f"{label.upper()} DETECTED", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    total_suspicious_actions += WEIGHT_CALCULATOR  # Tambahkan bobot untuk kalkulator/HP

        # Hitung persentase kecurangan
        cheating_percentage = (total_suspicious_actions / total_possible_actions) * 100

        # Kirim data ke frontend via WebSocket
        socketio.emit('update_data', {
            'turn_count': turn_count,
            'nod_count': nod_count,
            'cheating_percentage': round(cheating_percentage, 2)
        })

        # Encode frame ke JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    socketio.run(app, debug=True)