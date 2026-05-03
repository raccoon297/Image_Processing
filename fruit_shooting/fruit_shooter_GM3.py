"""
🍎 Hand Gun Fruit Shooter Game  v1.7 (최종 안정화 + 지터링 해결)
=====================================
수정 내용:
  - 과일 종류 편향 버그 수정 및 에임 감도 상승
  - 발사 로직: 까딱임(반동) 샷 및 과거 궤적 기억
  - 총 모양 인식 3D 업그레이드: 손가락이 정면을 향해도 인식 (원근 왜곡 해결)
  - [v1.7] 지터링(깜빡임) 완벽 해결: 히스테리시스(이중 임계값) + 프레임 지연 안정성 도입

실행:
    py -3.12 fruit_shooter.py
"""

import cv2
import mediapipe as mp
import pygame
import numpy as np
import random
import math
import time
import sys

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# ── 손 랜드마크 인덱스 ──
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_DIP  = 7;  INDEX_TIP  = 8
MIDDLE_MCP = 9;  MIDDLE_PIP = 10; MIDDLE_DIP = 11
RING_MCP   = 13; RING_PIP   = 14; RING_DIP   = 15
PINKY_MCP  = 17; PINKY_PIP  = 18; PINKY_DIP  = 19

# ── 화면/카메라 설정 ──
GAME_W, GAME_H = 1280, 720
CAM_W,  CAM_H  = 640,  480
FPS            = 60

# ── 손 → 화면 좌표 매핑 보정 (에임 감도) ──
CAM_PAD_X = 0.25   
CAM_PAD_Y = 0.20   

# ── 색상 ──
WHITE  = (255, 255, 255)
RED    = (220,  50,  50)
GREEN  = ( 50, 200,  80)
YELLOW = (255, 220,  30)
ORANGE = (255, 140,   0)
CYAN   = (  0, 220, 220)
GRAY   = (100, 100, 100)
DARK   = ( 20,  20,  30)

# ── 과일 설정 ──
FRUIT_COLORS = {
    "사과":  (200,  40,  40),
    "오렌지":(230, 120,  20),
    "레몬":  (240, 220,  30),
    "포도":  (130,  30, 160),
    "수박":  ( 50, 180,  60),
    "딸기":  (220,  30,  80),
}
FRUIT_NAMES  = list(FRUIT_COLORS.keys())
FRUIT_SCORES = {"사과": 10, "오렌지": 15, "레몬": 20,
                "포도": 25, "수박":   5,  "딸기": 30}

# ── 게임 파라미터 ──
SPAWN_INTERVAL  = 1.8    
FRUIT_R_MIN     = 45     
FRUIT_R_MAX     = 70     
FRUIT_SPEED_MIN = 60
FRUIT_SPEED_MAX = 110
BULLET_SPEED    = 1000
BULLET_R        = 12     
SHOOT_COOLDOWN  = 0.35
GAME_DURATION   = 60
CROSSHAIR_R     = 22     
CROSSHAIR_LINE  = 180    


# ──────────────────────────────────────────────
# 손 인식 헬퍼
# ──────────────────────────────────────────────

def finger_angle_3d(a, b, c):
    """MediaPipe의 x, y, z 3차원 좌표를 모두 사용하여 실제 각도 계산"""
    ab = (a.x - b.x, a.y - b.y, a.z - b.z)
    cb = (c.x - b.x, c.y - b.y, c.z - b.z)
    dot   = ab[0]*cb[0] + ab[1]*cb[1] + ab[2]*cb[2]
    mag_a = math.hypot(*ab) + 1e-6
    mag_c = math.hypot(*cb) + 1e-6
    return math.degrees(math.acos(max(-1.0, min(1.0, dot/(mag_a*mag_c)))))

def cam_to_game(nx, ny):
    nx = (nx - CAM_PAD_X) / (1.0 - 2 * CAM_PAD_X)
    ny = (ny - CAM_PAD_Y) / (1.0 - 2 * CAM_PAD_Y)
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return int(nx * GAME_W), int(ny * GAME_H)

def get_gun_tip_direction(lms, w, h):
    mcp, tip = lms[INDEX_MCP], lms[INDEX_TIP]
    dx, dy   = (tip.x - mcp.x) * w, (tip.y - mcp.y) * h
    dist     = math.hypot(dx, dy) + 1e-6
    return (dx/dist, dy/dist)


# ──────────────────────────────────────────────
# 게임 오브젝트
# ──────────────────────────────────────────────

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = float(x), float(y)
        self.color = color
        angle = random.uniform(0, math.tau)
        speed = random.uniform(100, 350)
        self.vx, self.vy = math.cos(angle)*speed, math.sin(angle)*speed
        self.life  = 1.0
        self.decay = random.uniform(1.2, 2.5)
        self.r     = random.randint(5, 14)

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 200 * dt
        self.life -= self.decay * dt

    def draw(self, surface):
        if self.life <= 0: return
        alpha  = max(0, int(self.life * 255))
        radius = max(1, int(self.r * self.life))
        s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
        r, g, b = self.color
        pygame.draw.circle(s, (r, g, b, alpha), (radius, radius), radius)
        surface.blit(s, (int(self.x)-radius, int(self.y)-radius))

    @property
    def alive(self): return self.life > 0

class Fruit:
    def __init__(self, gw, gh):
        self.name   = random.choice(FRUIT_NAMES)
        self.color  = FRUIT_COLORS[self.name]
        self.score  = FRUIT_SCORES[self.name]
        self.radius = random.randint(FRUIT_R_MIN, FRUIT_R_MAX)
        self.x      = float(random.randint(self.radius, gw-self.radius))
        self.y      = float(-self.radius)
        self.speed  = random.uniform(FRUIT_SPEED_MIN, FRUIT_SPEED_MAX)
        self.wamp   = random.uniform(0, 15)
        self.wspd   = random.uniform(1, 3)
        self.wt     = random.uniform(0, math.tau)
        self.alive  = True

    def update(self, dt, gh):
        self.y  += self.speed * dt
        self.wt += self.wspd * dt
        self.x  += math.sin(self.wt) * self.wamp * dt
        if self.y - self.radius > gh:
            self.alive = False

    def draw(self, surface, font):
        if not self.alive: return
        ix, iy = int(self.x), int(self.y)
        pygame.draw.circle(surface, self.color, (ix, iy), self.radius)
        pygame.draw.circle(surface, WHITE, (ix, iy), self.radius, 2)
        pygame.draw.circle(surface, WHITE,
                           (int(self.x - self.radius*0.28),
                            int(self.y - self.radius*0.28)),
                           max(5, self.radius // 3))
        txt = font.render(self.name, True, WHITE)
        surface.blit(txt, (ix - txt.get_width()//2, iy - txt.get_height()//2))

    def check_hit(self, bx, by):
        return math.hypot(bx-self.x, by-self.y) <= self.radius + BULLET_R

class Bullet:
    def __init__(self, x, y, dx, dy):
        self.x, self.y = float(x), float(y)
        self.dx, self.dy = dx, dy
        self.alive = True
        self.trail = []

    def update(self, dt, gw, gh):
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 10: self.trail.pop(0)
        self.x += self.dx * BULLET_SPEED * dt
        self.y += self.dy * BULLET_SPEED * dt
        if not (0 <= self.x <= gw and 0 <= self.y <= gh):
            self.alive = False

    def draw(self, surface):
        if not self.alive: return
        n = max(len(self.trail), 1)
        for i, (tx, ty) in enumerate(self.trail):
            alpha  = int(200 * (i+1)/n)
            radius = max(2, BULLET_R * (i+1)//n)
            s = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 255, 100, alpha), (radius, radius), radius)
            surface.blit(s, (tx-radius, ty-radius))
        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), BULLET_R)
        pygame.draw.circle(surface, WHITE,  (int(self.x), int(self.y)), BULLET_R//2)


# ──────────────────────────────────────────────
# 메인 게임
# ──────────────────────────────────────────────

class FruitShooterGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((GAME_W, GAME_H))
        pygame.display.set_caption("Hand Gun Fruit Shooter")
        self.clock  = pygame.time.Clock()

        self.font_lg  = pygame.font.SysFont("malgungothic", 52, bold=True)
        self.font_md  = pygame.font.SysFont("malgungothic", 32, bold=True)
        self.font_sm  = pygame.font.SysFont("malgungothic", 22)
        self.font_hud = pygame.font.SysFont("malgungothic", 28, bold=True)
        self.font_fr  = pygame.font.SysFont("malgungothic", 18, bold=True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

        self._init_landmarker()

        self.cam_frame    = None
        self.cam_aspect   = 4.0 / 3.0
        self.frame_ts_ms  = 0
        
        # ✅ 지터링 방지를 위한 상태 변수
        self.gun_detected       = False
        self.gun_valid_frames   = 0
        self.gun_invalid_frames = 0
        
        self.gun_tip      = (GAME_W//2, GAME_H//2)
        self.gun_dir      = (0, -1)
        self.trigger_on   = False
        self.smoothed_tip = None 
        self.tip_history  = [] 
        self.shot_origin  = None
        self.shot_dir     = None

        self.state       = "intro"
        self.score       = 0
        self.time_left   = GAME_DURATION
        self.start_time  = 0.0
        self.fruits      = []
        self.bullets     = []
        self.particles   = []
        self.last_spawn  = 0.0
        self.last_shot   = 0.0
        self.hit_effects = []

        rng = random.Random(42)
        self.stars = [(rng.randint(0,GAME_W), rng.randint(0,GAME_H),
                       rng.randint(1,3)) for _ in range(80)]

    def _init_landmarker(self):
        import urllib.request, os
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("손 인식 모델 다운로드 중... (최초 1회, 약 10MB)")
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/latest/"
                   "hand_landmarker.task")
            urllib.request.urlretrieve(url, model_path)
            print("다운로드 완료!")
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.65,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    # ── 카메라 처리 ──

    def process_camera(self):
        ret, frame = self.cap.read()
        if not ret: return

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        self.frame_ts_ms += int(1000 / FPS)
        result = self.landmarker.detect_for_video(mp_img, self.frame_ts_ms)

        raw_gun_shape = False

        # 1. 1차 모양 판별 (히스테리시스 적용)
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            idx_angle = finger_angle_3d(lms[INDEX_MCP], lms[INDEX_PIP], lms[INDEX_DIP])
            mid_angle = finger_angle_3d(lms[MIDDLE_MCP], lms[MIDDLE_PIP], lms[MIDDLE_DIP])

            if not self.gun_detected:
                # 총을 만들 때는 엄격하게 (140도 이상)
                raw_gun_shape = (idx_angle > 140 and mid_angle > 140)
            else:
                # 유지할 때는 너그럽게 (100도 이상이면 안 풀림)
                raw_gun_shape = (idx_angle > 100 and mid_angle > 100)

        # 2. 2차 상태 업데이트 (시간 기반 안정성: 프레임 지연)
        if raw_gun_shape:
            self.gun_invalid_frames = 0
            self.gun_valid_frames += 1
            if self.gun_valid_frames >= 3: # 3프레임(약 0.05초) 연속 인식 시 활성화
                self.gun_detected = True
        else:
            self.gun_valid_frames = 0
            self.gun_invalid_frames += 1
            if self.gun_invalid_frames >= 4: # 4프레임 연속 미인식 시 해제
                self.gun_detected = False

        # 3. 조준 및 발사 처리 (총이 안정적으로 인식된 상태일 때만)
        self.trigger_on = False

        if self.gun_detected and result.hand_landmarks:
            lms = result.hand_landmarks[0]
            tip_lm = lms[INDEX_TIP]
            gx, gy = cam_to_game(tip_lm.x, tip_lm.y)
            direction = get_gun_tip_direction(lms, CAM_W, CAM_H)

            # 선형 보간을 통한 손떨림 보정
            smooth_factor = 0.25 
            if self.smoothed_tip is None:
                self.smoothed_tip = (gx, gy)
            else:
                sx, sy = self.smoothed_tip
                self.smoothed_tip = (sx + smooth_factor * (gx - sx), sy + smooth_factor * (gy - sy))

            self.gun_tip = (int(self.smoothed_tip[0]), int(self.smoothed_tip[1]))
            self.gun_dir = direction

            # 발사 판별 (과거 궤적 기억)
            self.tip_history.append((self.gun_tip[0], self.gun_tip[1], direction[0], direction[1]))
            if len(self.tip_history) > 6:
                self.tip_history.pop(0)

            if len(self.tip_history) == 6:
                old_x, old_y, old_dx, old_dy = self.tip_history[0]
                new_x, new_y, _, _ = self.tip_history[-1]
                
                # 위로 까딱임 감지
                if (old_y - new_y) > 40:
                    self.trigger_on = True
                    self.shot_origin = (old_x, old_y)
                    self.shot_dir    = (old_dx, old_dy)
        else:
            # 총 인식이 풀리면 변수들 초기화
            self.smoothed_tip = None 
            self.tip_history.clear()

        # ── 카메라 화면 렌더링 준비 (손 골격 그리기) ──
        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            connections = [
                (0,1),(1,2),(2,3),(3,4),
                (0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),
                (9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),(0,17)
            ]
            pts = [(int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])) for lm in lms]
            for a, b in connections:
                cv2.line(frame, pts[a], pts[b], (0,200,0), 2)
            for pt in pts:
                cv2.circle(frame, pt, 5, (0,255,0), -1)
            if self.gun_detected:
                cv2.circle(frame, pts[INDEX_TIP], 10,
                           (0,0,255) if self.trigger_on else (0,255,255), 3)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb = np.ascontiguousarray(frame_rgb)
        cam_h, cam_w = frame_rgb.shape[:2]
        self.cam_aspect = cam_w / cam_h
        self.cam_frame = pygame.image.frombuffer(frame_rgb.tobytes(), (cam_w, cam_h), 'RGB')

    # ── 게임 로직 ──

    def update(self, dt):
        now = time.time()
        if self.state != "playing": return

        self.time_left = max(0, GAME_DURATION-(now-self.start_time))
        if self.time_left <= 0:
            self.state = "gameover"; return

        if now - self.last_spawn > SPAWN_INTERVAL:
            self.fruits.append(Fruit(GAME_W, GAME_H))
            self.last_spawn = now

        if (self.gun_detected and self.trigger_on and
                now - self.last_shot > SHOOT_COOLDOWN):
            self.bullets.append(Bullet(self.shot_origin[0], self.shot_origin[1],
                                       self.shot_dir[0], self.shot_dir[1]))
            self.last_shot = now
            self.tip_history.clear()

        for b in self.bullets: b.update(dt, GAME_W, GAME_H)
        self.bullets = [b for b in self.bullets if b.alive]

        for f in self.fruits:
            f.update(dt, GAME_H)
            for b in self.bullets:
                if b.alive and f.alive and f.check_hit(b.x, b.y):
                    f.alive = b.alive = False
                    self.score += f.score
                    for _ in range(20):
                        self.particles.append(Particle(f.x, f.y, f.color))
                    self.hit_effects.append([f.x, f.y, f'+{f.score}', 1.0])

        self.fruits    = [f for f in self.fruits    if f.alive]
        for p in self.particles: p.update(dt)
        self.particles = [p for p in self.particles if p.alive]
        for e in self.hit_effects: e[1] -= 55*dt; e[3] -= 1.5*dt
        self.hit_effects = [e for e in self.hit_effects if e[3] > 0]

    # ── 그리기 ──

    def draw_bg(self):
        self.screen.fill(DARK)
        for sx, sy, sr in self.stars:
            pygame.draw.circle(self.screen, (200,200,255), (sx,sy), sr)

    def draw_hud(self):
        self.screen.blit(
            self.font_hud.render(f"점수: {self.score}", True, YELLOW), (20,20))
        bw    = 300
        ratio = self.time_left / GAME_DURATION
        pygame.draw.rect(self.screen, GRAY, (GAME_W-bw-20,20,bw,28), border_radius=14)
        col = GREEN if ratio>.5 else (ORANGE if ratio>.25 else RED)
        pygame.draw.rect(self.screen, col,
                         (GAME_W-bw-20,20,int(bw*ratio),28), border_radius=14)
        tt = self.font_sm.render(f"{self.time_left:.1f}s", True, WHITE)
        self.screen.blit(tt, (GAME_W-bw//2-20-tt.get_width()//2, 23))
        col2 = GREEN if self.gun_detected else RED
        st   = "총 인식됨!" if self.gun_detected else "총 모양을 만드세요"
        self.screen.blit(self.font_sm.render(st, True, col2), (20, GAME_H-40))


    def draw_crosshair(self):
        if not self.gun_detected: return
        tx, ty = self.gun_tip
        dx, dy = self.gun_dir
        ex = int(tx + dx * CROSSHAIR_LINE)
        ey = int(ty + dy * CROSSHAIR_LINE)

        col = RED if self.trigger_on else YELLOW

        # pygame.draw.line(self.screen, col, (tx,ty), (ex,ey), 3)
        pygame.draw.circle(self.screen, col, (tx,ty), CROSSHAIR_R, 3)
        pygame.draw.circle(self.screen, col, (tx,ty), CROSSHAIR_R//3)
        pygame.draw.line(self.screen, col,
                         (tx-CROSSHAIR_R, ty), (tx+CROSSHAIR_R, ty), 2)
        pygame.draw.line(self.screen, col,
                         (tx, ty-CROSSHAIR_R), (tx, ty+CROSSHAIR_R), 2)

    
    def draw_cam(self):
        if self.cam_frame is None: return
        
        pw = 320
        ph = int(pw / self.cam_aspect)
        preview = pygame.transform.scale(self.cam_frame, (pw, ph))
        
        rx, ry  = GAME_W-pw-10, GAME_H-ph-10
        pygame.draw.rect(self.screen, CYAN,
                         (rx-3, ry-3, pw+6, ph+6), 3, border_radius=8)
        self.screen.blit(preview, (rx, ry))
        self.screen.blit(self.font_sm.render("카메라", True, CYAN), (rx, ry-28))

    def draw_hit_fx(self):
        for x, y, text, alpha in self.hit_effects:
            v = int(max(0, min(255, alpha*255)))
            s = self.font_md.render(text, True, (255, 220, v))
            self.screen.blit(s, (int(x)-s.get_width()//2, int(y)))

    def draw_intro(self):
        self.draw_bg()
        t = self.font_lg.render("Hand Gun Fruit Shooter", True, YELLOW)
        self.screen.blit(t, (GAME_W//2-t.get_width()//2, 130))
        for i, line in enumerate([
            "검지와 중지를 펴서 조준하세요 (나머지 손가락은 자유롭게!)",
            "총을 쥔 손을 위로 가볍게 '까딱' 올리면 발사됩니다!",
            "발사 시 위로 솟구치기 직전의 위치로 정확히 날아갑니다.",
            "화면 중앙 영역에서 살짝만 움직여도 끝까지 조준됩니다.",]):
            s = self.font_sm.render(line, True, WHITE)
            self.screen.blit(s, (GAME_W//2-s.get_width()//2, 270+i*40))

        for bx, col, label in [
            (GAME_W//2 - 280, GREEN,  "SPACE  게임 시작"),
            (GAME_W//2 + 20,  CYAN,   "  E       연습 모드"),
        ]:
            pygame.draw.rect(self.screen, col, (bx, 450, 240, 54), border_radius=12)
            pygame.draw.rect(self.screen, WHITE, (bx, 450, 240, 54), 2, border_radius=12)
            s = self.font_md.render(label, True, DARK)
            self.screen.blit(s, (bx + 240//2 - s.get_width()//2,
                                 450 + 54//2 - s.get_height()//2))

    def draw_gameover(self):
        self.draw_bg()
        ov = pygame.Surface((GAME_W,GAME_H), pygame.SRCALPHA)
        ov.fill((0,0,0,160)); self.screen.blit(ov,(0,0))
        g = self.font_lg.render("게임 오버!", True, RED)
        self.screen.blit(g, (GAME_W//2-g.get_width()//2, 220))
        s = self.font_md.render(f"최종 점수: {self.score}점", True, YELLOW)
        self.screen.blit(s, (GAME_W//2-s.get_width()//2, 320))
        r = self.font_sm.render("R 키로 다시 시작  /  ESC 종료", True, WHITE)
        self.screen.blit(r, (GAME_W//2-r.get_width()//2, 420))

    def draw_playing(self):
        self.draw_bg()
        for p in self.particles: p.draw(self.screen)
        for f in self.fruits:    f.draw(self.screen, self.font_fr)
        for b in self.bullets:   b.draw(self.screen)
        self.draw_hit_fx()
        self.draw_crosshair()
        self.draw_hud()
        self.draw_cam()

    # ── 연습 모드 로직 ──

    def update_practice(self, dt):
        now = time.time()
        
        if (self.gun_detected and self.trigger_on and
                now - self.last_shot > SHOOT_COOLDOWN):
            self.bullets.append(Bullet(self.shot_origin[0], self.shot_origin[1],
                                       self.shot_dir[0], self.shot_dir[1]))
            self.last_shot = now
            self.tip_history.clear()

        for b in self.bullets: b.update(dt, GAME_W, GAME_H)
        self.bullets = [b for b in self.bullets if b.alive]

        for t in self.practice_targets:
            if not t["alive"]:
                t["respawn_timer"] -= dt
                if t["respawn_timer"] <= 0:
                    t["alive"] = True
                continue

            for b in self.bullets:
                if b.alive and math.hypot(b.x-t["x"], b.y-t["y"]) <= t["radius"] + BULLET_R:
                    b.alive = False
                    t["alive"] = False
                    t["respawn_timer"] = 1.0
                    self.practice_hit_count += 1
                    for _ in range(22):
                        self.particles.append(Particle(t["x"], t["y"], t["color"]))
                    self.hit_effects.append([t["x"], t["y"], "HIT!", 1.0])

        for p in self.particles: p.update(dt)
        self.particles = [p for p in self.particles if p.alive]
        for e in self.hit_effects: e[1] -= 55*dt; e[3] -= 1.5*dt
        self.hit_effects = [e for e in self.hit_effects if e[3] > 0]

    def draw_practice(self):
        self.draw_bg()
        pygame.draw.line(self.screen, (60, 60, 80),
                         (0, GAME_H//2), (GAME_W, GAME_H//2), 1)

        for t in self.practice_targets:
            ix, iy = int(t["x"]), int(t["y"])
            if t["alive"]:
                pygame.draw.circle(self.screen, t["color"], (ix, iy), t["radius"])
                pygame.draw.circle(self.screen, WHITE, (ix, iy), t["radius"], 3)
                pygame.draw.circle(self.screen, WHITE,
                                   (int(t["x"]-t["radius"]*0.28),
                                    int(t["y"]-t["radius"]*0.28)),
                                   t["radius"]//3)
                txt = self.font_md.render(t["name"], True, WHITE)
                self.screen.blit(txt, (ix-txt.get_width()//2, iy-txt.get_height()//2))
                sc_txt = self.font_sm.render(f'{FRUIT_SCORES[t["name"]]}점', True, YELLOW)
                self.screen.blit(sc_txt, (ix-sc_txt.get_width()//2, iy+t["radius"]+8))
            else:
                pygame.draw.circle(self.screen, (60,60,60), (ix, iy), t["radius"], 3)
                wait_txt = self.font_sm.render("재생성 중...", True, GRAY)
                self.screen.blit(wait_txt, (ix-wait_txt.get_width()//2, iy-wait_txt.get_height()//2))

        for p in self.particles: p.draw(self.screen)
        self.draw_hit_fx()
        self.draw_crosshair()

        col = GREEN if self.gun_detected else RED
        st  = "✔ 총 인식됨!" if self.gun_detected else "✘ 검지와 중지를 펴세요"
        self.screen.blit(self.font_hud.render(st, True, col), (20, 20))

        trig_col = RED if self.trigger_on else GRAY
        trig_txt = "🔴 발사!" if self.trigger_on else "  대기 중"
        self.screen.blit(self.font_sm.render(trig_txt, True, trig_col), (20, 60))

        hit_txt = self.font_hud.render(f"맞힌 횟수: {self.practice_hit_count}", True, CYAN)
        self.screen.blit(hit_txt, (GAME_W//2 - hit_txt.get_width()//2, 20))

        esc_txt = self.font_sm.render("ESC  메뉴로 돌아가기", True, GRAY)
        self.screen.blit(esc_txt, (GAME_W - esc_txt.get_width() - 20, 20))

        title = self.font_sm.render("🎯  연습 모드 — 위로 까딱여서 발사하세요!", True, YELLOW)
        self.screen.blit(title, (GAME_W//2 - title.get_width()//2, GAME_H - 40))

        self.draw_cam()
        self.draw_hit_fx()

    # ── 상태 전환 ──

    def start_game(self):
        self.state      = "playing"
        self.score      = 0
        self.time_left  = GAME_DURATION
        self.start_time = self.last_spawn = time.time()
        self.fruits.clear(); self.bullets.clear()
        self.particles.clear(); self.hit_effects.clear()

    def start_practice(self):
        self.state = "practice"
        self.bullets.clear()
        self.particles.clear()
        self.hit_effects.clear()

        PRACTICE_FRUITS = [
            ("사과",  GAME_W // 4),
            ("레몬",  GAME_W // 2),
            ("포도",  GAME_W * 3 // 4),
        ]
        self.practice_targets = []
        for name, px in PRACTICE_FRUITS:
            self.practice_targets.append({
                "name":   name,
                "color":  FRUIT_COLORS[name],
                "x":      float(px),
                "y":      float(GAME_H // 2),
                "radius": 60,
                "alive":  True,
                "respawn_timer": 0.0,
            })
        self.practice_hit_count = 0

    def _quit(self):
        self.cap.release()
        self.landmarker.close()
        pygame.quit()
        sys.exit()

    # ── 메인 루프 ──

    def run(self):
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit()
                elif event.type == pygame.KEYDOWN:
                    if   event.key == pygame.K_ESCAPE:
                        if self.state == "practice": self.state = "intro"
                        else: self._quit()
                    elif event.key == pygame.K_SPACE  and self.state == "intro":    self.start_game()
                    elif event.key == pygame.K_e      and self.state == "intro":    self.start_practice()
                    elif event.key == pygame.K_r      and self.state == "gameover": self.start_game()

            self.process_camera()

            if   self.state == "playing":  self.update(dt)
            elif self.state == "practice": self.update_practice(dt)

            if   self.state == "intro":    self.draw_intro()
            elif self.state == "playing":  self.draw_playing()
            elif self.state == "practice": self.draw_practice()
            elif self.state == "gameover": self.draw_gameover()
            pygame.display.flip()


if __name__ == "__main__":
    game = FruitShooterGame()
    game.run()