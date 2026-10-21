import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os

# ==========================================
# 얼굴 인식 모델 준비
# ==========================================
MODEL_FILE = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/"
    "face_landmarker.task"
)

if not os.path.exists(MODEL_FILE):
    print("얼굴 인식 모델을 다운로드합니다...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_FILE)
    print("다운로드 완료!")

# ==========================================
# MediaPipe 설정
# ==========================================
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_FILE),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1
)

# 얼굴 특징점: 코끝, 턱, 양쪽 눈 바깥쪽, 양쪽 입꼬리
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]

# 얼굴의 일반적인 형태를 가정한 기준 좌표(mm).
# 사람마다 얼굴 형태가 달라 실제 측정값과 차이가 날 수 있습니다.
MODEL_POINTS = np.array([
    [0.0, 0.0, 0.0],
    [0.0, 63.0, -12.0],
    [-43.0, -32.0, -26.0],
    [43.0, -32.0, -26.0],
    [-28.0, 30.0, -20.0],
    [28.0, 30.0, -20.0]
], dtype=np.float64)


def calculate_pose(landmarks, width, height):
    image_points = np.array([
        [landmarks[i].x * width, landmarks[i].y * height]
        for i in LANDMARK_IDS
    ], dtype=np.float64)

    focal_length = float(width)
    camera_matrix = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    distortion = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles = cv2.RQDecomp3x3(rotation_matrix)[0]

    x = float(translation_vector[0][0])
    y = float(translation_vector[1][0])
    z = float(translation_vector[2][0])

    pitch = float(angles[0])
    roll = float(angles[2])

    return x, y, z, pitch, roll, image_points


def main():
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("카메라를 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")
        return

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                ret, frame = camera.read()
                if not ret:
                    print("카메라 영상을 읽을 수 없습니다.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = landmarker.detect(mp_image)

                cv2.drawMarker(
                    frame, (width // 2, height // 2),
                    (255, 255, 0), cv2.MARKER_CROSS, 20, 2
                )

                if result.face_landmarks:
                    # num_faces=1 설정으로 얼굴 하나만 추적합니다.
                    landmarks = result.face_landmarks[0]
                    pose = calculate_pose(landmarks, width, height)

                    if pose is not None:
                        x, y, z, pitch, roll, points = pose

                        for px, py in points.astype(int):
                            cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

                        xs = [int(p.x * width) for p in landmarks]
                        ys = [int(p.y * height) for p in landmarks]
                        cv2.rectangle(
                            frame,
                            (min(xs), min(ys)),
                            (max(xs), max(ys)),
                            (0, 255, 0), 2
                        )

                        lines = [
                            f"X: {x:.1f} mm",
                            f"Y: {y:.1f} mm",
                            f"Z: {z:.1f} mm (estimated)",
                            f"Pitch: {pitch:.1f} deg",
                            f"Roll: {roll:.1f} deg"
                        ]

                        for i, line in enumerate(lines):
                            cv2.putText(
                                frame, line, (20, 35 + i * 32),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0), 2
                            )

                        # Z는 근사값입니다. 350mm 기준은 정확히 보정된 거리가 아닙니다.
                        if 0 < z < 350:
                            cv2.putText(
                                frame, "WARNING: TOO CLOSE!",
                                (20, height - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                                (0, 0, 255), 3
                            )
                else:
                    cv2.putText(
                        frame, "No face detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
                    )

                cv2.imshow("Face Tracking App", frame)

                # ESC 키로 종료
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
