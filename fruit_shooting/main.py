"""Hand Gun Fruit Shooter 실행 진입점."""

from fruit_shooter.game import FruitShooterGame


def main() -> None:
    game = FruitShooterGame()
    game.run()


if __name__ == "__main__":
    main()
