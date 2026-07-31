"""게임 상태, 충돌 처리, 렌더링, 메인 루프."""

import math
import random
import time

import pygame

from .config import (
    BULLET_R,
    CAMERA_PREVIEW_W,
    CROSSHAIR_R,
    CYAN,
    DARK,
    FPS,
    FRUIT_COLORS,
    FRUIT_SCORES,
    GAME_DURATION,
    GAME_H,
    GAME_W,
    GRAY,
    GREEN,
    ORANGE,
    RED,
    SHOOT_COOLDOWN,
    SPAWN_INTERVAL,
    WHITE,
    YELLOW,
)
from .entities import Bullet, Fruit, Particle
from .hand_tracker import HandTracker


class FruitShooterGame:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((GAME_W, GAME_H))
        pygame.display.set_caption("Hand Gun Fruit Shooter")
        self.clock = pygame.time.Clock()

        self.font_lg = pygame.font.SysFont("malgungothic", 52, bold=True)
        self.font_md = pygame.font.SysFont("malgungothic", 32, bold=True)
        self.font_sm = pygame.font.SysFont("malgungothic", 22)
        self.font_hud = pygame.font.SysFont("malgungothic", 28, bold=True)
        self.font_fruit = pygame.font.SysFont("malgungothic", 18, bold=True)

        self.hand = HandTracker()
        self.running = True
        self.state = "intro"

        self.score = 0
        self.time_left = float(GAME_DURATION)
        self.start_time = 0.0
        self.last_spawn = 0.0
        self.last_shot = 0.0

        self.fruits: list[Fruit] = []
        self.bullets: list[Bullet] = []
        self.particles: list[Particle] = []
        self.hit_effects: list[list[float | str]] = []
        self.practice_targets: list[dict[str, object]] = []
        self.practice_hit_count = 0

        random_generator = random.Random(42)
        self.stars = [
            (
                random_generator.randint(0, GAME_W),
                random_generator.randint(0, GAME_H),
                random_generator.randint(1, 3),
            )
            for _ in range(80)
        ]

    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(FPS) / 1000.0
                self._handle_events()
                if not self.running:
                    break

                self.hand.update()

                if self.state == "playing":
                    self._update_game(dt)
                elif self.state == "practice":
                    self._update_practice(dt)

                self._draw_current_state()
                pygame.display.flip()
        finally:
            self.close()

    def close(self) -> None:
        self.hand.close()
        pygame.quit()

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                continue

            if event.type != pygame.KEYDOWN:
                continue

            if event.key == pygame.K_ESCAPE:
                if self.state == "practice":
                    self.state = "intro"
                else:
                    self.running = False
            elif event.key == pygame.K_SPACE and self.state == "intro":
                self._start_game()
            elif event.key == pygame.K_e and self.state == "intro":
                self._start_practice()
            elif event.key == pygame.K_r and self.state == "gameover":
                self._start_game()

    def _update_game(self, dt: float) -> None:
        now = time.time()
        self.time_left = max(0.0, GAME_DURATION - (now - self.start_time))
        if self.time_left <= 0:
            self.state = "gameover"
            return

        if now - self.last_spawn > SPAWN_INTERVAL:
            self.fruits.append(Fruit(GAME_W))
            self.last_spawn = now

        self._try_fire(now)
        self._update_bullets(dt)
        self._update_fruits(dt)
        self._update_effects(dt)

    def _update_practice(self, dt: float) -> None:
        now = time.time()
        self._try_fire(now)
        self._update_bullets(dt)

        for target in self.practice_targets:
            if not bool(target["alive"]):
                target["respawn_timer"] = float(target["respawn_timer"]) - dt
                if float(target["respawn_timer"]) <= 0:
                    target["alive"] = True
                continue

            for bullet in self.bullets:
                distance = math.hypot(
                    bullet.x - float(target["x"]),
                    bullet.y - float(target["y"]),
                )
                if bullet.alive and distance <= float(target["radius"]) + BULLET_R:
                    bullet.alive = False
                    target["alive"] = False
                    target["respawn_timer"] = 1.0
                    self.practice_hit_count += 1
                    self._spawn_hit_effect(
                        float(target["x"]),
                        float(target["y"]),
                        target["color"],
                        "HIT!",
                        particle_count=22,
                    )

        self._update_effects(dt)

    def _try_fire(self, now: float) -> None:
        if not self.hand.gun_detected or now - self.last_shot <= SHOOT_COOLDOWN:
            return

        shot = self.hand.consume_shot()
        if shot is None:
            return

        origin, direction = shot
        self.bullets.append(Bullet(origin[0], origin[1], direction[0], direction[1]))
        self.last_shot = now

    def _update_bullets(self, dt: float) -> None:
        for bullet in self.bullets:
            bullet.update(dt, GAME_W, GAME_H)
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]

    def _update_fruits(self, dt: float) -> None:
        for fruit in self.fruits:
            fruit.update(dt, GAME_H)
            for bullet in self.bullets:
                if bullet.alive and fruit.alive and fruit.check_hit(bullet.x, bullet.y):
                    fruit.alive = False
                    bullet.alive = False
                    self.score += fruit.score
                    self._spawn_hit_effect(
                        fruit.x,
                        fruit.y,
                        fruit.color,
                        f"+{fruit.score}",
                    )

        self.fruits = [fruit for fruit in self.fruits if fruit.alive]

    def _spawn_hit_effect(
        self,
        x: float,
        y: float,
        color: tuple[int, int, int],
        text: str,
        particle_count: int = 20,
    ) -> None:
        for _ in range(particle_count):
            self.particles.append(Particle(x, y, color))
        self.hit_effects.append([x, y, text, 1.0])

    def _update_effects(self, dt: float) -> None:
        for particle in self.particles:
            particle.update(dt)
        self.particles = [particle for particle in self.particles if particle.alive]

        for effect in self.hit_effects:
            effect[1] = float(effect[1]) - 55 * dt
            effect[3] = float(effect[3]) - 1.5 * dt
        self.hit_effects = [effect for effect in self.hit_effects if float(effect[3]) > 0]

    def _draw_current_state(self) -> None:
        if self.state == "intro":
            self._draw_intro()
        elif self.state == "playing":
            self._draw_playing()
        elif self.state == "practice":
            self._draw_practice()
        elif self.state == "gameover":
            self._draw_gameover()

    def _draw_background(self) -> None:
        self.screen.fill(DARK)
        for star_x, star_y, star_radius in self.stars:
            pygame.draw.circle(
                self.screen,
                (200, 200, 255),
                (star_x, star_y),
                star_radius,
            )

    def _draw_hud(self) -> None:
        score_text = self.font_hud.render(f"점수: {self.score}", True, YELLOW)
        self.screen.blit(score_text, (20, 20))

        bar_width = 300
        ratio = self.time_left / GAME_DURATION
        bar_x = GAME_W - bar_width - 20
        pygame.draw.rect(
            self.screen,
            GRAY,
            (bar_x, 20, bar_width, 28),
            border_radius=14,
        )
        bar_color = GREEN if ratio > 0.5 else ORANGE if ratio > 0.25 else RED
        pygame.draw.rect(
            self.screen,
            bar_color,
            (bar_x, 20, int(bar_width * ratio), 28),
            border_radius=14,
        )

        time_text = self.font_sm.render(f"{self.time_left:.1f}s", True, WHITE)
        self.screen.blit(
            time_text,
            (bar_x + bar_width // 2 - time_text.get_width() // 2, 23),
        )

        status_color = GREEN if self.hand.gun_detected else RED
        status = "총 인식됨!" if self.hand.gun_detected else "총 모양을 만드세요"
        self.screen.blit(
            self.font_sm.render(status, True, status_color),
            (20, GAME_H - 40),
        )

    def _draw_crosshair(self) -> None:
        if not self.hand.gun_detected:
            return

        x, y = self.hand.aim_position
        color = RED if self.hand.trigger_on else YELLOW
        pygame.draw.circle(self.screen, color, (x, y), CROSSHAIR_R, 3)
        pygame.draw.circle(self.screen, color, (x, y), CROSSHAIR_R // 3)
        pygame.draw.line(
            self.screen,
            color,
            (x - CROSSHAIR_R, y),
            (x + CROSSHAIR_R, y),
            2,
        )
        pygame.draw.line(
            self.screen,
            color,
            (x, y - CROSSHAIR_R),
            (x, y + CROSSHAIR_R),
            2,
        )

    def _draw_camera(self) -> None:
        if self.hand.camera_surface is None:
            return

        preview_width = CAMERA_PREVIEW_W
        preview_height = int(preview_width / self.hand.camera_aspect)
        preview = pygame.transform.scale(
            self.hand.camera_surface,
            (preview_width, preview_height),
        )

        x = GAME_W - preview_width - 10
        y = GAME_H - preview_height - 10
        pygame.draw.rect(
            self.screen,
            CYAN,
            (x - 3, y - 3, preview_width + 6, preview_height + 6),
            3,
            border_radius=8,
        )
        self.screen.blit(preview, (x, y))
        self.screen.blit(self.font_sm.render("카메라", True, CYAN), (x, y - 28))

    def _draw_hit_effects(self) -> None:
        for x, y, text, alpha in self.hit_effects:
            brightness = int(max(0, min(255, float(alpha) * 255)))
            rendered = self.font_md.render(str(text), True, (255, 220, brightness))
            self.screen.blit(
                rendered,
                (int(float(x)) - rendered.get_width() // 2, int(float(y))),
            )

    def _draw_intro(self) -> None:
        self._draw_background()
        title = self.font_lg.render("Hand Gun Fruit Shooter", True, YELLOW)
        self.screen.blit(title, (GAME_W // 2 - title.get_width() // 2, 130))

        instructions = (
            "검지와 중지를 펴서 조준하세요 (나머지 손가락은 자유롭게!)",
            "총을 쥔 손을 위로 가볍게 '까딱' 올리면 발사됩니다!",
            "발사 시 위로 솟구치기 직전의 위치로 정확히 날아갑니다.",
            "화면 중앙 영역에서 살짝만 움직여도 끝까지 조준됩니다.",
        )
        for index, line in enumerate(instructions):
            rendered = self.font_sm.render(line, True, WHITE)
            self.screen.blit(
                rendered,
                (GAME_W // 2 - rendered.get_width() // 2, 270 + index * 40),
            )

        buttons = (
            (GAME_W // 2 - 280, GREEN, "SPACE  게임 시작"),
            (GAME_W // 2 + 20, CYAN, "  E       연습 모드"),
        )
        for x, color, label in buttons:
            pygame.draw.rect(self.screen, color, (x, 450, 240, 54), border_radius=12)
            pygame.draw.rect(
                self.screen,
                WHITE,
                (x, 450, 240, 54),
                2,
                border_radius=12,
            )
            rendered = self.font_md.render(label, True, DARK)
            self.screen.blit(
                rendered,
                (
                    x + 120 - rendered.get_width() // 2,
                    477 - rendered.get_height() // 2,
                ),
            )


    def _draw_gameover(self) -> None:
        self._draw_background()
        overlay = pygame.Surface((GAME_W, GAME_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        title = self.font_lg.render("게임 오버!", True, RED)
        self.screen.blit(title, (GAME_W // 2 - title.get_width() // 2, 220))
        score = self.font_md.render(f"최종 점수: {self.score}점", True, YELLOW)
        self.screen.blit(score, (GAME_W // 2 - score.get_width() // 2, 320))
        restart = self.font_sm.render("R 키로 다시 시작  /  ESC 종료", True, WHITE)
        self.screen.blit(restart, (GAME_W // 2 - restart.get_width() // 2, 420))

    def _draw_playing(self) -> None:
        self._draw_background()
        for particle in self.particles:
            particle.draw(self.screen)
        for fruit in self.fruits:
            fruit.draw(self.screen, self.font_fruit)
        for bullet in self.bullets:
            bullet.draw(self.screen)
        self._draw_hit_effects()
        self._draw_crosshair()
        self._draw_hud()
        self._draw_camera()

    def _draw_practice(self) -> None:
        self._draw_background()
        pygame.draw.line(
            self.screen,
            (60, 60, 80),
            (0, GAME_H // 2),
            (GAME_W, GAME_H // 2),
            1,
        )

        for target in self.practice_targets:
            x = int(float(target["x"]))
            y = int(float(target["y"]))
            radius = int(target["radius"])

            if bool(target["alive"]):
                pygame.draw.circle(self.screen, target["color"], (x, y), radius)
                pygame.draw.circle(self.screen, WHITE, (x, y), radius, 3)
                pygame.draw.circle(
                    self.screen,
                    WHITE,
                    (
                        int(float(target["x"]) - radius * 0.28),
                        int(float(target["y"]) - radius * 0.28),
                    ),
                    radius // 3,
                )
                name = str(target["name"])
                name_text = self.font_md.render(name, True, WHITE)
                self.screen.blit(
                    name_text,
                    (x - name_text.get_width() // 2, y - name_text.get_height() // 2),
                )
                score_text = self.font_sm.render(f"{FRUIT_SCORES[name]}점", True, YELLOW)
                self.screen.blit(
                    score_text,
                    (x - score_text.get_width() // 2, y + radius + 8),
                )
            else:
                pygame.draw.circle(self.screen, (60, 60, 60), (x, y), radius, 3)
                waiting = self.font_sm.render("재생성 중...", True, GRAY)
                self.screen.blit(
                    waiting,
                    (x - waiting.get_width() // 2, y - waiting.get_height() // 2),
                )

        for particle in self.particles:
            particle.draw(self.screen)
        self._draw_hit_effects()
        self._draw_crosshair()

        status_color = GREEN if self.hand.gun_detected else RED
        status = "✔ 총 인식됨!" if self.hand.gun_detected else "✘ 검지와 중지를 펴세요"
        self.screen.blit(self.font_hud.render(status, True, status_color), (20, 20))

        trigger_color = RED if self.hand.trigger_on else GRAY
        trigger_text = "🔴 발사!" if self.hand.trigger_on else "  대기 중"
        self.screen.blit(self.font_sm.render(trigger_text, True, trigger_color), (20, 60))

        hits = self.font_hud.render(f"맞힌 횟수: {self.practice_hit_count}", True, CYAN)
        self.screen.blit(hits, (GAME_W // 2 - hits.get_width() // 2, 20))

        back = self.font_sm.render("ESC  메뉴로 돌아가기", True, GRAY)
        self.screen.blit(back, (GAME_W - back.get_width() - 20, 20))

        title = self.font_sm.render("🎯  연습 모드 — 위로 까딱여서 발사하세요!", True, YELLOW)
        self.screen.blit(title, (GAME_W // 2 - title.get_width() // 2, GAME_H - 40))
        self._draw_camera()

    def _start_game(self) -> None:
        self.state = "playing"
        self.score = 0
        self.time_left = float(GAME_DURATION)
        self.start_time = time.time()
        self.last_spawn = self.start_time
        self.fruits.clear()
        self.bullets.clear()
        self.particles.clear()
        self.hit_effects.clear()

    def _start_practice(self) -> None:
        self.state = "practice"
        self.bullets.clear()
        self.particles.clear()
        self.hit_effects.clear()

        practice_fruits = (
            ("사과", GAME_W // 4),
            ("레몬", GAME_W // 2),
            ("포도", GAME_W * 3 // 4),
        )
        self.practice_targets = [
            {
                "name": name,
                "color": FRUIT_COLORS[name],
                "x": float(x),
                "y": float(GAME_H // 2),
                "radius": 60,
                "alive": True,
                "respawn_timer": 0.0,
            }
            for name, x in practice_fruits
        ]
        self.practice_hit_count = 0
