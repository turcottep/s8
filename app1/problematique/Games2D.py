import heapq
import itertools
import math
import time
from pygame.locals import *
import pygame

from Player import *
from Maze import *
from Constants import *
from fuzzy import get_movement_for_ai_fuzzy
from genetics_main import train_genetics
from astar import astar
import numpy as np

from swiplserver import PrologMQI


class App:
    windowWidth = WIDTH
    windowHeight = HEIGHT
    player = 0

    def __init__(self, mazefile):
        self._running = True
        self._win = False
        self._dead = False
        self._display_surf = None
        self._image_surf = None
        self._block_surf = None
        self._clock = None
        self.level = 0
        self.score = 0
        self.timer = 0.0
        self.player = Player()
        self.historical_player_path = []
        self.maze = Maze(mazefile)
        self.ai_mode = True

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(
            (self.windowWidth, self.windowHeight), pygame.HWSURFACE
        )
        self._clock = pygame.time.Clock()
        pygame.display.set_caption("Dungeon Crawler")
        pygame.time.set_timer(pygame.USEREVENT, 10)
        self._running = True
        self.maze.make_maze_wall_list()
        self.maze.make_maze_item_lists()
        self._image_surf = pygame.image.load("assets/kickboxeuse.png")
        self.player.set_position(
            1.5 * self.maze.tile_size_x, 0.5 * self.maze.tile_size_y
        )
        self.player.set_size(
            PLAYER_SIZE * self.maze.tile_size_x, PLAYER_SIZE * self.maze.tile_size_x
        )
        self._image_surf = pygame.transform.scale(
            self._image_surf, self.player.get_size()
        )
        self._block_surf = pygame.image.load("assets/wall.png")

    def on_keyboard_input(self, keys):
        if keys[K_RIGHT] or keys[K_d]:
            self.move_player_right()

        if keys[K_LEFT] or keys[K_a]:
            self.move_player_left()

        if keys[K_UP] or keys[K_w]:
            self.move_player_up()

        if keys[K_DOWN] or keys[K_s]:
            self.move_player_down()

        if keys[K_k]:
            print("k")
            self.ai_mode = False

        if keys[K_l]:
            print("l")
            self.ai_mode = True

        # Utility functions for AI
        if keys[K_p]:
            print("perception:")
            print(self.maze.make_perception_list(self.player, self._display_surf))
            print("")
            # returns a list of 4 lists of pygame.rect inside the perception radius
            # the 4 lists are [wall_list, obstacle_list, item_list, monster_list]
            # item_list includes coins and treasure

        if keys[K_m]:
            print("mock fight...:")
            for i, monster in enumerate(self.maze.monsterList):
                print("monster:", i, monster.mock_fight(self.player))
            # returns the number of rounds you win against the monster
            # you need to win all four rounds to beat it

        if keys[K_ESCAPE]:
            self._running = False

    # FONCTION ?? Ajuster selon votre format d'instruction
    def on_AI_input(self, instruction):
        # print("instruction", instruction)
        # "instruction" est une liste de 4 ??l??ments [up, right, down, left]
        if self.ai_mode:
            if instruction[0]:
                self.move_player_up()

            if instruction[1]:
                self.move_player_right()

            if instruction[2]:
                self.move_player_down()

            if instruction[3]:
                self.move_player_left()

    def move_player_right(self):
        self.player.moveRight()
        if self.on_wall_collision() or self.on_obstacle_collision():
            self.player.moveLeft()

    def move_player_left(self):
        self.player.moveLeft()
        if self.on_wall_collision() or self.on_obstacle_collision():
            self.player.moveRight()

    def move_player_up(self):
        self.player.moveUp()
        if self.on_wall_collision() or self.on_obstacle_collision():
            self.player.moveDown()

    def move_player_down(self):
        self.player.moveDown()
        if self.on_wall_collision() or self.on_obstacle_collision():
            self.player.moveUp()

    def on_wall_collision(self):
        collide_index = self.player.get_rect().collidelist(self.maze.wallList)
        if not collide_index == -1:
            # print("Collision Detected!")
            return True
        return False

    def on_obstacle_collision(self):
        collide_index = self.player.get_rect().collidelist(self.maze.obstacleList)
        if not collide_index == -1:
            # print("Collision Detected!")
            return True
        return False

    def on_coin_collision(self):
        collide_index = self.player.get_rect().collidelist(self.maze.coinList)
        if not collide_index == -1:
            self.maze.coinList.pop(collide_index)
            return True
        else:
            return False

    def on_treasure_collision(self):
        collide_index = self.player.get_rect().collidelist(self.maze.treasureList)
        if not collide_index == -1:
            self.maze.treasureList.pop(collide_index)
            return True
        else:
            return False

    def on_monster_collision(self):
        for monster in self.maze.monsterList:
            if self.player.get_rect().colliderect(monster.rect):
                return monster
        return False

    def on_exit(self):
        return self.player.get_rect().colliderect(self.maze.exit)

    def maze_render(self):
        self._display_surf.fill((0, 0, 0))
        self.maze.draw(self._display_surf, self._block_surf)
        font = pygame.font.SysFont(None, 32)
        text = font.render("Coins: " + str(self.score), True, BLACK)
        self._display_surf.blit(text, (WIDTH - 120, 10))
        text = font.render("Time: " + format(self.timer, ".2f"), True, BLACK)
        self._display_surf.blit(text, (WIDTH - 300, 10))

    def on_render(self, path, path_simplified):
        self.maze_render()

        for historical_player in self.historical_player_path:
            pygame.draw.rect(
                self._display_surf,
                (100, 100, 0),
                (
                    historical_player[0],
                    historical_player[1],
                    self.player.size_x,
                    self.player.size_y,
                ),
            )

        pygame.draw.rect(
            self._display_surf,
            (200, 100, 0),
            (
                self.player.x,
                self.player.y,
                self.player.size_x,
                self.player.size_y,
            ),
        )
        self.historical_player_path.append(
            (
                self.player.x,
                self.player.y,
            )
        )

        for i in range(len(path) - 1):
            live_cell = path[i]
            next_cell = path[i + 1]

            pygame.draw.line(
                self._display_surf,
                (0, 0, 255),
                (
                    (live_cell[0] + 0.5) * self.maze.tile_size_x,
                    (live_cell[1] + 0.5) * self.maze.tile_size_y,
                ),
                (
                    (next_cell[0] + 0.5) * self.maze.tile_size_x,
                    (next_cell[1] + 0.5) * self.maze.tile_size_y,
                ),
                2,
            )

        for i in range(len(path_simplified) - 1):
            live_cell = path_simplified[i]
            next_cell = path_simplified[i + 1]

            pygame.draw.line(
                self._display_surf,
                (255, 100, 255),
                (
                    (live_cell[0] + 0.5) * self.maze.tile_size_x + 2,
                    (live_cell[1] + 0.5) * self.maze.tile_size_y + 2,
                ),
                (
                    (next_cell[0] + 0.5) * self.maze.tile_size_x + 2,
                    (next_cell[1] + 0.5) * self.maze.tile_size_y + 2,
                ),
                2,
            )

        self._display_surf.blit(self._image_surf, (self.player.x, self.player.y))
        pygame.display.flip()

    def on_win_render(self):
        self.maze_render()
        font = pygame.font.SysFont(None, 120)
        text = font.render("CONGRATULATIONS!", True, GREEN)
        self._display_surf.blit(text, (0.1 * self.windowWidth, 0.4 * self.windowHeight))
        pygame.display.flip()

    def on_death_render(self):
        self.maze_render()
        font = pygame.font.SysFont(None, 120)
        text = font.render("YOU DIED!", True, RED)
        self._display_surf.blit(text, (0.1 * self.windowWidth, 0.4 * self.windowHeight))
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        self.on_init()

        best_attributes = do_genetics(self.player, self.maze.monsterList)
        [path, path_simplified] = do_planification(self.maze)
        original_monster_list = []
        for monster in self.maze.monsterList:
            original_monster_list.append(monster)
        step_index = 0

        while self._running:
            self._clock.tick(GAME_CLOCK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._running = False
                if event.type == pygame.USEREVENT:
                    self.timer += 0.01
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            self.on_keyboard_input(keys)
            # print("player size", self.player.get_rect())
            if len(original_monster_list) > 0:
                check_for_incoming_monster_to_set_stats(
                    player=self.player,
                    monster_list=original_monster_list,
                    best_attributes=best_attributes,
                )

            # [key_to_press_for_AI, step_index] = get_movement_for_ai(
            #     path,
            #     self.player,
            #     self.maze.make_perception_list(self.player, self._display_surf),
            #     step_index,
            #     self.maze,
            # )

            [key_to_press_for_AI, step_index] = get_movement_for_ai_fuzzy(
                path,
                self.player.get_position(),
                self.maze.make_perception_list(self.player, self._display_surf),
                step_index,
                self.player.get_rect()[2],
                self.maze.tile_size_x,
                obs_size=8,
            )

            self.on_AI_input(key_to_press_for_AI)  # A d??commenter pour utiliser l'IA
            if self.on_coin_collision():
                self.score += 1
            if self.on_treasure_collision():
                self.score += 10
            monster = self.on_monster_collision()
            if monster:
                if monster.fight(self.player):
                    self.maze.monsterList.remove(monster)
                else:
                    self._running = False
                    self._dead = True
            if self.on_exit():
                self._running = False
                self._win = True
            self.on_render(path, path_simplified)

        while self._win:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._win = False
            self.on_win_render()

        while self._dead:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._dead = False
            self.on_death_render()

        self.on_cleanup()


def do_planification_static(maze):

    path_simplified = [
        (1, 0),
        (1, 5),
        (2, 5),
        (2, 6),
        (5, 10),
        (6, 1),
        (7, 1),
        (8, 1),
        (10, 1),
        (9, 4),
        (9, 3),
        (10, 3),
        (10, 4),
        (9, 8),
        (7, 10),
        (7, 16),
        (7, 20),
        (5, 17),
        (5, 16),
        (3, 14),
        (1, 14),
        (1, 11),
        (1, 10),
        (1, 9),
        (1, 8),
        (1, 7),
        (5, 12),
        (1, 18),
        (3, 19),
        (3, 20),
        (5, 20),
        (4, 21),
        (4, 22),
        (5, 22),
        (1, 21),
        (1, 22),
        (2, 22),
        (12, 10),
        (12, 18),
        (9, 16),
        (10, 16),
        (10, 17),
        (15, 1),
        (17, 1),
        (17, 2),
        (19, 2),
        (19, 1),
        (21, 1),
        (22, 1),
        (29, 4),
        (29, 6),
        (30, 6),
        (29, 8),
        (30, 8),
        (29, 10),
        (29, 13),
        (24, 9),
        (24, 6),
        (18, 6),
        (16, 7),
        (16, 10),
        (18, 15),
        (14, 11),
        (14, 10),
        (14, 7),
        (14, 6),
        (16, 4),
        (19, 4),
        (14, 16),
        (18, 17),
        (16, 17),
        (16, 16),
        (16, 15),
        (28, 22),
        (27, 21),
        (25, 21),
        (23, 21),
        (21, 21),
        (20, 22),
        (22, 22),
        (24, 22),
        (26, 22),
        (16, 22),
        (16, 21),
        (14, 21),
        (14, 22),
        (22, 11),
    ]
    path = [
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (1, 5),
        (2, 5),
        (2, 6),
        (3, 6),
        (4, 6),
        (5, 6),
        (5, 7),
        (5, 8),
        (5, 9),
        (5, 10),
        (5, 9),
        (5, 8),
        (5, 7),
        (5, 6),
        (6, 6),
        (6, 5),
        (6, 4),
        (6, 3),
        (6, 2),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (9, 1),
        (8, 1),
        (7, 1),
        (6, 1),
        (6, 2),
        (6, 3),
        (6, 4),
        (7, 4),
        (8, 4),
        (9, 4),
        (9, 3),
        (10, 3),
        (10, 4),
        (9, 4),
        (8, 4),
        (7, 4),
        (6, 4),
        (6, 5),
        (6, 6),
        (7, 6),
        (8, 6),
        (9, 6),
        (9, 7),
        (9, 8),
        (9, 9),
        (9, 10),
        (8, 10),
        (7, 10),
        (7, 11),
        (7, 12),
        (7, 13),
        (7, 14),
        (7, 15),
        (7, 16),
        (7, 17),
        (7, 18),
        (7, 19),
        (7, 20),
        (7, 19),
        (7, 18),
        (6, 18),
        (5, 18),
        (5, 17),
        (5, 16),
        (5, 17),
        (5, 18),
        (6, 18),
        (7, 18),
        (7, 17),
        (7, 16),
        (7, 15),
        (7, 14),
        (6, 14),
        (5, 14),
        (4, 14),
        (3, 14),
        (2, 14),
        (1, 14),
        (1, 13),
        (1, 12),
        (1, 11),
        (1, 10),
        (1, 9),
        (1, 8),
        (1, 7),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (1, 12),
        (2, 12),
        (3, 12),
        (4, 12),
        (5, 12),
        (4, 12),
        (3, 12),
        (2, 12),
        (1, 12),
        (1, 13),
        (1, 14),
        (2, 14),
        (3, 14),
        (3, 15),
        (3, 16),
        (2, 16),
        (1, 16),
        (1, 17),
        (1, 18),
        (2, 18),
        (2, 19),
        (3, 19),
        (3, 20),
        (4, 20),
        (5, 20),
        (4, 20),
        (4, 21),
        (4, 22),
        (5, 22),
        (4, 22),
        (4, 21),
        (4, 20),
        (3, 20),
        (2, 20),
        (2, 21),
        (1, 21),
        (1, 22),
        (2, 22),
        (2, 21),
        (2, 20),
        (2, 19),
        (2, 18),
        (1, 18),
        (1, 17),
        (1, 16),
        (2, 16),
        (3, 16),
        (3, 15),
        (3, 14),
        (4, 14),
        (5, 14),
        (6, 14),
        (7, 14),
        (7, 13),
        (8, 13),
        (9, 13),
        (10, 13),
        (11, 13),
        (12, 13),
        (12, 12),
        (12, 11),
        (12, 10),
        (12, 11),
        (12, 12),
        (12, 13),
        (12, 14),
        (12, 15),
        (12, 16),
        (12, 17),
        (12, 18),
        (12, 19),
        (11, 19),
        (10, 19),
        (9, 19),
        (9, 18),
        (9, 17),
        (9, 16),
        (10, 16),
        (10, 17),
        (9, 17),
        (9, 18),
        (9, 19),
        (10, 19),
        (11, 19),
        (12, 19),
        (12, 18),
        (12, 17),
        (12, 16),
        (12, 15),
        (12, 14),
        (12, 13),
        (12, 12),
        (12, 11),
        (12, 10),
        (12, 9),
        (12, 8),
        (11, 8),
        (10, 8),
        (9, 8),
        (9, 7),
        (9, 6),
        (10, 6),
        (11, 6),
        (12, 6),
        (12, 5),
        (12, 4),
        (12, 3),
        (12, 2),
        (13, 2),
        (14, 2),
        (15, 2),
        (15, 1),
        (16, 1),
        (17, 1),
        (17, 2),
        (18, 2),
        (19, 2),
        (19, 1),
        (20, 1),
        (21, 1),
        (22, 1),
        (22, 2),
        (23, 2),
        (24, 2),
        (25, 2),
        (26, 2),
        (27, 2),
        (28, 2),
        (29, 2),
        (30, 2),
        (30, 3),
        (30, 4),
        (29, 4),
        (28, 4),
        (28, 5),
        (28, 6),
        (29, 6),
        (30, 6),
        (29, 6),
        (28, 6),
        (28, 7),
        (28, 8),
        (29, 8),
        (30, 8),
        (29, 8),
        (28, 8),
        (28, 9),
        (28, 10),
        (29, 10),
        (30, 10),
        (30, 11),
        (30, 12),
        (30, 13),
        (29, 13),
        (28, 13),
        (27, 13),
        (26, 13),
        (25, 13),
        (24, 13),
        (24, 12),
        (24, 11),
        (24, 10),
        (24, 9),
        (24, 8),
        (24, 7),
        (24, 6),
        (23, 6),
        (22, 6),
        (21, 6),
        (20, 6),
        (19, 6),
        (18, 6),
        (17, 6),
        (16, 6),
        (16, 7),
        (16, 8),
        (16, 9),
        (16, 10),
        (16, 9),
        (16, 8),
        (16, 7),
        (16, 6),
        (17, 6),
        (18, 6),
        (19, 6),
        (20, 6),
        (21, 6),
        (22, 6),
        (23, 6),
        (24, 6),
        (24, 7),
        (24, 8),
        (24, 9),
        (24, 10),
        (24, 11),
        (24, 12),
        (24, 13),
        (23, 13),
        (22, 13),
        (21, 13),
        (20, 13),
        (19, 13),
        (18, 13),
        (18, 14),
        (18, 15),
        (18, 14),
        (18, 13),
        (17, 13),
        (16, 13),
        (15, 13),
        (14, 13),
        (14, 12),
        (14, 11),
        (14, 10),
        (14, 9),
        (14, 8),
        (14, 7),
        (14, 6),
        (14, 5),
        (14, 4),
        (15, 4),
        (16, 4),
        (17, 4),
        (18, 4),
        (19, 4),
        (18, 4),
        (17, 4),
        (16, 4),
        (15, 4),
        (14, 4),
        (14, 5),
        (14, 6),
        (14, 7),
        (14, 8),
        (14, 9),
        (14, 10),
        (14, 11),
        (14, 12),
        (14, 13),
        (14, 14),
        (14, 15),
        (14, 16),
        (14, 17),
        (14, 18),
        (14, 19),
        (15, 19),
        (16, 19),
        (17, 19),
        (18, 19),
        (19, 19),
        (20, 19),
        (21, 19),
        (22, 19),
        (23, 19),
        (24, 19),
        (25, 19),
        (25, 18),
        (25, 17),
        (24, 17),
        (23, 17),
        (22, 17),
        (21, 17),
        (20, 17),
        (19, 17),
        (18, 17),
        (17, 17),
        (16, 17),
        (16, 16),
        (16, 15),
        (16, 16),
        (16, 17),
        (17, 17),
        (18, 17),
        (19, 17),
        (20, 17),
        (20, 16),
        (20, 15),
        (21, 15),
        (22, 15),
        (23, 15),
        (24, 15),
        (25, 15),
        (26, 15),
        (27, 15),
        (28, 15),
        (29, 15),
        (30, 15),
        (30, 16),
        (30, 17),
        (30, 18),
        (30, 19),
        (30, 20),
        (30, 21),
        (30, 22),
        (29, 22),
        (28, 22),
        (27, 22),
        (27, 21),
        (26, 21),
        (25, 21),
        (24, 21),
        (23, 21),
        (22, 21),
        (21, 21),
        (20, 21),
        (20, 22),
        (21, 22),
        (22, 22),
        (23, 22),
        (24, 22),
        (25, 22),
        (26, 22),
        (25, 22),
        (24, 22),
        (23, 22),
        (22, 22),
        (21, 22),
        (20, 22),
        (19, 22),
        (18, 22),
        (17, 22),
        (16, 22),
        (16, 21),
        (15, 21),
        (14, 21),
        (14, 22),
        (14, 21),
        (15, 21),
        (16, 21),
        (17, 21),
        (18, 21),
        (19, 21),
        (20, 21),
        (21, 21),
        (22, 21),
        (23, 21),
        (24, 21),
        (25, 21),
        (26, 21),
        (27, 21),
        (28, 21),
        (28, 20),
        (28, 19),
        (27, 19),
        (27, 18),
        (27, 17),
        (28, 17),
        (29, 17),
        (30, 17),
        (30, 16),
        (30, 15),
        (29, 15),
        (28, 15),
        (27, 15),
        (26, 15),
        (25, 15),
        (24, 15),
        (23, 15),
        (22, 15),
        (21, 15),
        (20, 15),
        (20, 16),
        (20, 17),
        (21, 17),
        (22, 17),
        (23, 17),
        (24, 17),
        (25, 17),
        (25, 18),
        (25, 19),
        (24, 19),
        (23, 19),
        (22, 19),
        (21, 19),
        (20, 19),
        (19, 19),
        (18, 19),
        (17, 19),
        (16, 19),
        (15, 19),
        (14, 19),
        (14, 18),
        (14, 17),
        (14, 16),
        (14, 15),
        (14, 14),
        (14, 13),
        (15, 13),
        (16, 13),
        (17, 13),
        (18, 13),
        (19, 13),
        (20, 13),
        (21, 13),
        (22, 13),
        (23, 13),
        (24, 13),
        (24, 12),
        (24, 11),
        (24, 10),
        (24, 9),
        (24, 8),
        (24, 7),
        (24, 6),
        (23, 6),
        (22, 6),
        (21, 6),
        (20, 6),
        (19, 6),
        (18, 6),
        (17, 6),
        (16, 6),
        (16, 7),
        (16, 8),
        (16, 9),
        (16, 10),
        (16, 11),
        (17, 11),
        (18, 11),
        (19, 11),
        (20, 11),
        (20, 10),
        (20, 9),
        (20, 8),
        (21, 8),
        (22, 8),
        (22, 9),
        (22, 10),
        (22, 11),
    ]

    return [path, path_simplified]


def do_planification(maze):
    # print("maze", maze.maze)

    # start timer
    start_time = time.time()

    # # A-star
    starting_positon = None
    ending_position = None
    points_to_visit = []

    for j, row in enumerate(maze.maze):
        for i, cell in enumerate(row):
            if cell == "S":
                starting_positon = (i, j)
            elif cell == "E":
                ending_position = (i, j)
            elif cell == "C" or cell == "T":
                points_to_visit.append((i, j))

    # print("starting_positon", starting_positon)
    # print("ending_position", ending_position)
    # print("points_to_visit", points_to_visit)

    # travelling salesman problem
    # start from starting_positon, visit all points_to_visit, end at ending_position
    # find the shortest path

    # print("weigth_matrix", weigth_matrix)
    all_points = [starting_positon] + points_to_visit + [ending_position]
    weigth_matrix = brushfire(all_points)

    # find the shortest path
    path_simplified_index = shortest_path(weigth_matrix)
    path_simplified = []
    # print("path_simplified_index", path_simplified_index)
    # print("path_indexlen", len(path_simplified_index))

    path = [starting_positon]
    current_position = starting_positon
    for i in range(len(all_points)):
        point_index = path_simplified_index[i]
        point = all_points[point_index]
        path_simplified.append(point)
        print(i, "current_position", current_position, "point", point, end="\r")
        path_temp = astar(current_position, point)
        # add to path except the first one
        path = path + path_temp[1:]
        # print("path", path)
        current_position = point

    # print("path simplified", path_simplified)
    # print("path", path)

    # raise Exception("stop")

    # end timer
    end_time = time.time()
    print("time for path_planning", end_time - start_time, "seconds")

    return [path, path_simplified]


class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f


def brushfire(all_points):

    # return distances from all_points to all_points

    weigth_matrix = np.zeros((len(all_points), len(all_points)))

    with PrologMQI() as mqi:
        with PrologMQI() as mqi_file:
            with mqi_file.create_thread() as prolog_thread:

                result = prolog_thread.query("[prolog/maze].")
                # print("[prolog/maze].", result)

                result = prolog_thread.query("[prolog/functions].")
                # print("[prolog/functions].", result)

                # fill the matrix with the taxi distance
                for i, point1 in enumerate(all_points):

                    print(
                        i,
                        "/",
                        len(all_points),
                        "calculating distances from",
                        point1,
                        end="\r",
                    )
                    # for i in range(1):
                    point1 = all_points[i]

                    # brushfire algorithm to find the distance to every point

                    # Initialize both open and closed list
                    open_list = []
                    closed_list = []

                    # Heapify the open_list and Add the start node
                    heapq.heapify(open_list)

                    heapq.heappush(open_list, Node(None, point1))

                    while len(open_list) > 0:

                        current_node = heapq.heappop(open_list)
                        # print("current_node is", current_node)

                        closed_list.append(current_node)

                        query_str = (
                            "passage("
                            + str(current_node.position[0])
                            + ", "
                            + str(current_node.position[1])
                            + ", X, Y)."
                        )

                        all_moves = prolog_thread.query(query_str)
                        # print("all_moves", all_moves)
                        moves_clean = []
                        for move in all_moves:
                            moves_clean.append((move["X"], move["Y"]))

                        # print("moves_clean", moves_clean)

                        for move in moves_clean:

                            # if child is not in the closed list
                            if (
                                len(
                                    [
                                        closed_child
                                        for closed_child in closed_list
                                        if closed_child.position == move
                                    ]
                                )
                                > 0
                            ):
                                continue

                            # calculate f value
                            child = Node(current_node, move)
                            child.f = current_node.f + 1

                            # Child is already in the open list
                            if (
                                len(
                                    [
                                        open_node
                                        for open_node in open_list
                                        if child.position == open_node.position
                                        and child.f > open_node.f
                                    ]
                                )
                                > 0
                            ):
                                continue

                            heapq.heappush(open_list, child)

                    # print("closed_list", closed_list)

                    for node in closed_list:
                        if node.position in all_points:
                            weigth_matrix[i, all_points.index(node.position)] = node.f

                return weigth_matrix


def shortest_path(weigth_matrix):

    # find the next closest point to visit
    path = [0]
    current_position = 0
    for i in range(weigth_matrix.shape[0] - 1):
        print("current_position", current_position, end="\r")
        # find the closest point
        min_distance = 100000
        min_index = 0
        for index in range(weigth_matrix.shape[0] - 1):
            if index not in path:
                distance = weigth_matrix[current_position, index]
                if distance < min_distance:
                    min_distance = distance
                    min_index = index
        # print("min_index", min_index)
        path.append(min_index)
        current_position = min_index

    # add the last one
    # path.append(len(weigth_matrix))
    # remove last one
    path_clean = path[:-1]
    # add the last one
    path_clean.append(len(weigth_matrix) - 1)
    return path_clean


def check_for_incoming_monster_to_set_stats(player, monster_list, best_attributes):
    # set stats to beat the closest monster

    min_distance = 1000000000
    monster_index = 0
    # print("monster_list", monster_list)
    for i, monster in enumerate(monster_list):
        monster_x = monster.rect.x
        monster_y = monster.rect.y
        player_x = player.x
        player_y = player.y
        # print(
        #     "player_x",
        #     player_x,
        #     "player_y",
        #     player_y,
        #     "monster_x",
        #     monster_x,
        #     "monster_y",
        #     monster_y,
        # )
        distance = (monster_x - player_x) ** 2 + (monster_y - player_y) ** 2
        if distance < min_distance:
            min_distance = distance
            monster_index = i
    player.set_attributes(best_attributes[monster_index])


def genetics_fitness_function(player, monster, attributes):
    # calculate the fitness for the attributes
    fitness = np.zeros((attributes.shape[0], 2))
    best_score = 0
    for i in range(attributes.shape[0]):
        attributes_i = attributes[i, :]

        # print("attributes_i", attributes_i)
        for j, attribute in enumerate(attributes_i):
            if attribute < -MAX_ATTRIBUTE:
                print("attribute", attribute, "is too low")
            if attribute > MAX_ATTRIBUTE:
                print("attribute", attribute, "is too high")

        player.set_attributes(attributes_i)
        results = monster.mock_fight(player)
        # print("results", results)
        fitness[i, :] = results
        if results[0] > best_score:
            best_score = results[0]
    return fitness


def do_genetics(player, monsterList):

    # return a list of the best attributes for each monster

    # starting timer
    start = time.time()

    best_attributes = []

    monster_index = 0
    while len(best_attributes) < len(monsterList):
        monster = monsterList[monster_index]
        print(
            "training genetics to fight monster: ",
            monster_index + 1,
            " of ",
            len(monsterList),
        )
        best_genetics = train_genetics(
            lambda x: genetics_fitness_function(player, monster, x),
            NUM_ATTRIBUTES,
            -MAX_ATTRIBUTE,
            MAX_ATTRIBUTE,
            4,
        )
        player.set_attributes(best_genetics)
        results = monster.mock_fight(player)
        print("results best_attributes", results)
        if results[0] == 4:
            best_attributes.append(best_genetics)
            monster_index += 1

    # end timer
    end = time.time()
    print("time to train genetics", end - start, "seconds")

    return best_attributes
