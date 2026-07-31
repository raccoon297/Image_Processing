"""게임과 손 인식에 사용하는 공통 설정값."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/"
    "hand_landmarker.task"
)

# 화면과 카메라
GAME_W, GAME_H = 1280, 720
CAM_W, CAM_H = 640, 480
FPS = 60

# 카메라 좌표를 게임 화면에 매핑할 때 사용할 유효 영역
CAM_PAD_X = 0.25
CAM_PAD_Y = 0.20

# 손 랜드마크 인덱스
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11

# 제스처 안정화
GUN_ACTIVATE_ANGLE = 140
GUN_HOLD_ANGLE = 100
GUN_ACTIVATE_FRAMES = 3
GUN_RELEASE_FRAMES = 4
AIM_SMOOTH_FACTOR = 0.25
SHOT_HISTORY_SIZE = 6
SHOT_FLICK_DISTANCE = 40

# 색상
WHITE = (255, 255, 255)
RED = (220, 50, 50)
GREEN = (50, 200, 80)
YELLOW = (255, 220, 30)
ORANGE = (255, 140, 0)
CYAN = (0, 220, 220)
GRAY = (100, 100, 100)
DARK = (20, 20, 30)

# 과일
FRUIT_COLORS = {
    "사과": (200, 40, 40),
    "오렌지": (230, 120, 20),
    "레몬": (240, 220, 30),
    "포도": (130, 30, 160),
    "수박": (50, 180, 60),
    "딸기": (220, 30, 80),
}
FRUIT_NAMES = tuple(FRUIT_COLORS)
FRUIT_SCORES = {
    "사과": 10,
    "오렌지": 15,
    "레몬": 20,
    "포도": 25,
    "수박": 5,
    "딸기": 30,
}

# 게임 파라미터
SPAWN_INTERVAL = 1.8
FRUIT_R_MIN = 45
FRUIT_R_MAX = 70
FRUIT_SPEED_MIN = 60
FRUIT_SPEED_MAX = 110
BULLET_SPEED = 1000
BULLET_R = 12
SHOOT_COOLDOWN = 0.35
GAME_DURATION = 60
CROSSHAIR_R = 22
CAMERA_PREVIEW_W = 320
