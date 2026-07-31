"""과일, 총알, 파티클처럼 게임 화면에 등장하는 객체."""

import math
import random

import pygame

from .config import (
    BULLET_R,
    BULLET_SPEED,
    FRUIT_COLORS,
    FRUIT_NAMES,
    FRUIT_R_MAX,
    FRUIT_R_MIN,
    FRUIT_SCORES,
    FRUIT_SPEED_MAX,
    FRUIT_SPEED_MIN,
    WHITE,
    YELLOW,
)


class Particle:
    def __init__(self, x: float, y: float, color: tuple[int, int, int]) -> None:
        self.x = float(x)
        self.y = float(y)
        self.color = color

        angle = random.uniform(0, math.tau)
        speed = random.uniform(100, 350)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = 1.0
        self.decay = random.uniform(1.2, 2.5)
        self.radius = random.randint(5, 14)

    def update(self, dt: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 200 * dt
        self.life -= self.decay * dt

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        alpha = max(0, int(self.life * 255))
        radius = max(1, int(self.radius * self.life))
        particle_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        red, green, blue = self.color
        pygame.draw.circle(
            particle_surface,
            (red, green, blue, alpha),
            (radius, radius),
            radius,
        )
        surface.blit(particle_surface, (int(self.x) - radius, int(self.y) - radius))

    @property
    def alive(self) -> bool:
        return self.life > 0


class Fruit:
    def __init__(self, game_width: int) -> None:
        self.name = random.choice(FRUIT_NAMES)
        self.color = FRUIT_COLORS[self.name]
        self.score = FRUIT_SCORES[self.name]
        self.radius = random.randint(FRUIT_R_MIN, FRUIT_R_MAX)
        self.x = float(random.randint(self.radius, game_width - self.radius))
        self.y = float(-self.radius)
        self.speed = random.uniform(FRUIT_SPEED_MIN, FRUIT_SPEED_MAX)
        self.wave_amplitude = random.uniform(0, 15)
        self.wave_speed = random.uniform(1, 3)
        self.wave_time = random.uniform(0, math.tau)
        self.alive = True

    def update(self, dt: float, game_height: int) -> None:
        self.y += self.speed * dt
        self.wave_time += self.wave_speed * dt
        self.x += math.sin(self.wave_time) * self.wave_amplitude * dt

        if self.y - self.radius > game_height:
            self.alive = False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        if not self.alive:
            return

        x, y = int(self.x), int(self.y)
        pygame.draw.circle(surface, self.color, (x, y), self.radius)
        pygame.draw.circle(surface, WHITE, (x, y), self.radius, 2)
        pygame.draw.circle(
            surface,
            WHITE,
            (
                int(self.x - self.radius * 0.28),
                int(self.y - self.radius * 0.28),
            ),
            max(5, self.radius // 3),
        )

        text = font.render(self.name, True, WHITE)
        surface.blit(text, (x - text.get_width() // 2, y - text.get_height() // 2))

    def check_hit(self, bullet_x: float, bullet_y: float) -> bool:
        distance = math.hypot(bullet_x - self.x, bullet_y - self.y)
        return distance <= self.radius + BULLET_R


class Bullet:
    def __init__(self, x: float, y: float, dx: float, dy: float) -> None:
        self.x = float(x)
        self.y = float(y)
        self.dx = dx
        self.dy = dy
        self.alive = True
        self.trail: list[tuple[int, int]] = []

    def update(self, dt: float, game_width: int, game_height: int) -> None:
        self.trail.append((int(self.x), int(self.y)))
        if len(self.trail) > 10:
            self.trail.pop(0)

        self.x += self.dx * BULLET_SPEED * dt
        self.y += self.dy * BULLET_SPEED * dt

        if not (0 <= self.x <= game_width and 0 <= self.y <= game_height):
            self.alive = False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.alive:
            return

        trail_length = max(len(self.trail), 1)
        for index, (trail_x, trail_y) in enumerate(self.trail):
            alpha = int(200 * (index + 1) / trail_length)
            radius = max(2, BULLET_R * (index + 1) // trail_length)
            trail_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(
                trail_surface,
                (255, 255, 100, alpha),
                (radius, radius),
                radius,
            )
            surface.blit(trail_surface, (trail_x - radius, trail_y - radius))

        pygame.draw.circle(surface, YELLOW, (int(self.x), int(self.y)), BULLET_R)
        pygame.draw.circle(surface, WHITE, (int(self.x), int(self.y)), BULLET_R // 2)
