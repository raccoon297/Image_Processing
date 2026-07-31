"""웹캠 입력, 손 모양 인식, 조준 및 발사 제스처 처리."""

import math
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pygame
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from .config import (
    AIM_SMOOTH_FACTOR,
    CAM_H,
    CAM_PAD_X,
    CAM_PAD_Y,
    CAM_W,
    FPS,
    GAME_H,
    GAME_W,
    GUN_ACTIVATE_ANGLE,
    GUN_ACTIVATE_FRAMES,
    GUN_HOLD_ANGLE,
    GUN_RELEASE_FRAMES,
    INDEX_DIP,
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_DIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MODEL_PATH,
    MODEL_URL,
    SHOT_FLICK_DISTANCE,
    SHOT_HISTORY_SIZE,
)

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def finger_angle_3d(point_a, point_b, point_c) -> float:
    """MediaPipe의 x, y, z 좌표로 관절 각도를 계산한다."""
    vector_ab = (
        point_a.x - point_b.x,
        point_a.y - point_b.y,
        point_a.z - point_b.z,
    )
    vector_cb = (
        point_c.x - point_b.x,
        point_c.y - point_b.y,
        point_c.z - point_b.z,
    )

    dot_product = sum(a * b for a, b in zip(vector_ab, vector_cb))
    magnitude_ab = math.sqrt(sum(value * value for value in vector_ab)) + 1e-6
    magnitude_cb = math.sqrt(sum(value * value for value in vector_cb)) + 1e-6
    cosine = max(-1.0, min(1.0, dot_product / (magnitude_ab * magnitude_cb)))
    return math.degrees(math.acos(cosine))


def camera_to_game(normalized_x: float, normalized_y: float) -> tuple[int, int]:
    """카메라 정규화 좌표를 게임 화면 좌표로 변환한다."""
    x = (normalized_x - CAM_PAD_X) / (1.0 - 2 * CAM_PAD_X)
    y = (normalized_y - CAM_PAD_Y) / (1.0 - 2 * CAM_PAD_Y)
    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))
    return int(x * GAME_W), int(y * GAME_H)


def get_gun_direction(landmarks) -> tuple[float, float]:
    """검지 MCP에서 검지 끝으로 향하는 단위 방향 벡터를 계산한다."""
    mcp = landmarks[INDEX_MCP]
    tip = landmarks[INDEX_TIP]
    dx = (tip.x - mcp.x) * CAM_W
    dy = (tip.y - mcp.y) * CAM_H
    distance = math.hypot(dx, dy) + 1e-6
    return dx / distance, dy / distance


class HandTracker:
    """손 입력과 카메라 미리보기 상태를 한곳에서 관리한다."""

    def __init__(self, camera_index: int = 0) -> None:
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        if not self.capture.isOpened():
            raise RuntimeError("웹캠을 열 수 없습니다. 카메라 연결과 권한을 확인하세요.")

        self.landmarker = self._create_landmarker()
        self.frame_timestamp_ms = 0

        self.camera_surface: pygame.Surface | None = None
        self.camera_aspect = 4.0 / 3.0

        self.gun_detected = False
        self.gun_valid_frames = 0
        self.gun_invalid_frames = 0

        self.aim_position = (GAME_W // 2, GAME_H // 2)
        self.aim_direction = (0.0, -1.0)
        self.trigger_on = False
        self.shot_origin: tuple[int, int] | None = None
        self.shot_direction: tuple[float, float] | None = None

        self._smoothed_tip: tuple[float, float] | None = None
        self._tip_history: list[tuple[int, int, float, float]] = []

    @staticmethod
    def _create_landmarker() -> HandLandmarker:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not MODEL_PATH.exists():
            print("손 인식 모델 다운로드 중... (최초 1회, 약 10MB)")
            urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
            print("다운로드 완료!")

        base_options = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        options = HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        return HandLandmarker.create_from_options(options)

    def update(self) -> bool:
        """카메라 한 프레임을 처리한다. 성공하면 True를 반환한다."""
        success, frame = self.capture.read()
        if not success:
            self.trigger_on = False
            return False

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mediapipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        self.frame_timestamp_ms += int(1000 / FPS)
        result = self.landmarker.detect_for_video(
            mediapipe_image,
            self.frame_timestamp_ms,
        )

        landmarks = result.hand_landmarks[0] if result.hand_landmarks else None
        self._update_gun_state(landmarks)
        self._update_aim_and_trigger(landmarks)
        self._draw_hand_landmarks(frame, landmarks)
        self._update_camera_surface(frame)
        return True

    def consume_shot(self) -> tuple[tuple[int, int], tuple[float, float]] | None:
        """현재 발사 정보를 반환하고 궤적을 비운다."""
        if not self.trigger_on or self.shot_origin is None or self.shot_direction is None:
            return None

        shot = self.shot_origin, self.shot_direction
        self._tip_history.clear()
        return shot

    def close(self) -> None:
        self.capture.release()
        self.landmarker.close()

    def _update_gun_state(self, landmarks) -> None:
        raw_gun_shape = False

        if landmarks:
            index_angle = finger_angle_3d(
                landmarks[INDEX_MCP],
                landmarks[INDEX_PIP],
                landmarks[INDEX_DIP],
            )
            middle_angle = finger_angle_3d(
                landmarks[MIDDLE_MCP],
                landmarks[MIDDLE_PIP],
                landmarks[MIDDLE_DIP],
            )
            threshold = GUN_HOLD_ANGLE if self.gun_detected else GUN_ACTIVATE_ANGLE
            raw_gun_shape = index_angle > threshold and middle_angle > threshold

        if raw_gun_shape:
            self.gun_invalid_frames = 0
            self.gun_valid_frames += 1
            if self.gun_valid_frames >= GUN_ACTIVATE_FRAMES:
                self.gun_detected = True
        else:
            self.gun_valid_frames = 0
            self.gun_invalid_frames += 1
            if self.gun_invalid_frames >= GUN_RELEASE_FRAMES:
                self.gun_detected = False

    def _update_aim_and_trigger(self, landmarks) -> None:
        self.trigger_on = False
        self.shot_origin = None
        self.shot_direction = None

        if not self.gun_detected or landmarks is None:
            self._smoothed_tip = None
            self._tip_history.clear()
            return

        tip_landmark = landmarks[INDEX_TIP]
        game_x, game_y = camera_to_game(tip_landmark.x, tip_landmark.y)
        direction = get_gun_direction(landmarks)

        if self._smoothed_tip is None:
            self._smoothed_tip = float(game_x), float(game_y)
        else:
            smooth_x, smooth_y = self._smoothed_tip
            self._smoothed_tip = (
                smooth_x + AIM_SMOOTH_FACTOR * (game_x - smooth_x),
                smooth_y + AIM_SMOOTH_FACTOR * (game_y - smooth_y),
            )

        self.aim_position = (
            int(self._smoothed_tip[0]),
            int(self._smoothed_tip[1]),
        )
        self.aim_direction = direction

        self._tip_history.append(
            (
                self.aim_position[0],
                self.aim_position[1],
                direction[0],
                direction[1],
            )
        )
        if len(self._tip_history) > SHOT_HISTORY_SIZE:
            self._tip_history.pop(0)

        if len(self._tip_history) == SHOT_HISTORY_SIZE:
            old_x, old_y, old_dx, old_dy = self._tip_history[0]
            _, new_y, _, _ = self._tip_history[-1]
            if old_y - new_y > SHOT_FLICK_DISTANCE:
                self.trigger_on = True
                self.shot_origin = old_x, old_y
                self.shot_direction = old_dx, old_dy

    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks) -> None:
        if landmarks is None:
            return

        points = [
            (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
            for landmark in landmarks
        ]

        for start, end in _HAND_CONNECTIONS:
            cv2.line(frame, points[start], points[end], (0, 200, 0), 2)
        for point in points:
            cv2.circle(frame, point, 5, (0, 255, 0), -1)

        if self.gun_detected:
            color = (0, 0, 255) if self.trigger_on else (0, 255, 255)
            cv2.circle(frame, points[INDEX_TIP], 10, color, 3)

    def _update_camera_surface(self, frame: np.ndarray) -> None:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = np.ascontiguousarray(rgb_frame)
        camera_height, camera_width = rgb_frame.shape[:2]
        self.camera_aspect = camera_width / camera_height
        self.camera_surface = pygame.image.frombuffer(
            rgb_frame.tobytes(),
            (camera_width, camera_height),
            "RGB",
        )
