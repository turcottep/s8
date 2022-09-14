import pygame
import csv
from Monster import *
from Constants import *


def write_maze_to_prolog(maze):
    with open("prolog/maze.pl", "w") as file:
        for j, row in enumerate(maze):
            for i, value in enumerate(row):
                if value == "1":
                    value = 1
                else:
                    value = 0
                file.write(f"maze({i}, {j}, {value}).\n")


class Maze:
    def __init__(self, mazefile):
        self.tile_size_x = 0
        self.tile_size_y = 0
        self.maze = []
        with open(mazefile) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                self.maze.append(row)
        write_maze_to_prolog(self.maze)
        self.N = len(self.maze)
        self.M = len(self.maze[0])
        self.wallList = []
        self.coinList = []
        self.treasureList = []
        self.obstacleList = []
        self.monsterList = []
        self.exit = []
        self.tile_size_x = WIDTH / self.M
        self.tile_size_y = HEIGHT / self.N
        self._coin_surf = pygame.image.load("assets/coin.png")
        self._coin_surf = pygame.transform.scale(
            self._coin_surf,
            (ITEM_SIZE * self.tile_size_x, ITEM_SIZE * self.tile_size_y),
        )
        self._treasure_surf = pygame.image.load("assets/treasure.png")
        self._treasure_surf = pygame.transform.scale(
            self._treasure_surf,
            (2 * ITEM_SIZE * self.tile_size_x, 2 * ITEM_SIZE * self.tile_size_y),
        )
        self._monster_surf = pygame.image.load("assets/monster1.png")
        self._monster_surf = pygame.transform.scale(
            self._monster_surf, (self.tile_size_x, self.tile_size_y)
        )

    def random_position(self, i, j):
        x = (j + random.uniform(0, 1 - ITEM_SIZE)) * self.tile_size_x
        y = (i + random.uniform(0, 1 - ITEM_SIZE)) * self.tile_size_y
        return x, y

    def make_maze_wall_list(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == "1":
                    cell = pygame.Rect(
                        (j * self.tile_size_x, i * self.tile_size_y),
                        (self.tile_size_x, self.tile_size_y),
                    )
                    self.wallList.append(cell)

    def make_maze_item_lists(self):
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if self.maze[i][j] == COIN:
                    new_coin = pygame.Rect(
                        (self.random_position(i, j)),
                        (ITEM_SIZE * self.tile_size_x, ITEM_SIZE * self.tile_size_y),
                    )
                    self.coinList.append(new_coin)
                elif self.maze[i][j] == TREASURE:
                    new_treasure = pygame.Rect(
                        (self.random_position(i, j)),
                        (ITEM_SIZE * self.tile_size_x, ITEM_SIZE * self.tile_size_y),
                    )
                    self.treasureList.append(new_treasure)
                elif self.maze[i][j] == OBSTACLE:
                    new_obstacle = pygame.Rect(
                        (self.random_position(i, j)),
                        (ITEM_SIZE * self.tile_size_x, ITEM_SIZE * self.tile_size_y),
                    )
                    self.obstacleList.append(new_obstacle)
                elif self.maze[i][j] == MONSTER:
                    new_monster = pygame.Rect(
                        (j * self.tile_size_x, i * self.tile_size_y),
                        (self.tile_size_x, self.tile_size_y),
                    )
                    self.monsterList.append(Monster(new_monster))
                elif self.maze[i][j] == EXIT:
                    self.exit = pygame.Rect(
                        (j * self.tile_size_x, i * self.tile_size_y),
                        (self.tile_size_x, self.tile_size_y),
                    )

    def draw(self, display_surf, image_surf):
        image_surf = pygame.transform.scale(
            image_surf, (self.tile_size_x, self.tile_size_y)
        )
        for i in range(len(self.maze)):
            for j in range(len(self.maze[i])):
                if i % 2 == 0 and j % 2 == 0:
                    color_test = (150, 150, 150)
                elif i % 2 == 0 and j % 2 == 1:
                    color_test = (100, 100, 100)
                elif i % 2 == 1 and j % 2 == 0:
                    color_test = (100, 100, 100)
                elif i % 2 == 1 and j % 2 == 1:
                    color_test = (150, 150, 150)
                pygame.draw.rect(
                    display_surf,
                    color_test,
                    (
                        j * self.tile_size_x,
                        i * self.tile_size_y,
                        self.tile_size_x,
                        self.tile_size_y,
                    ),
                )
                if self.maze[i][j] == "1":
                    display_surf.blit(
                        image_surf, (j * self.tile_size_x, i * self.tile_size_y)
                    )
                elif self.maze[i][j] == "S":
                    pygame.draw.rect(
                        display_surf,
                        BLUE,
                        (
                            j * self.tile_size_x,
                            i * self.tile_size_y,
                            self.tile_size_x,
                            self.tile_size_y,
                        ),
                    )
                elif self.maze[i][j] == "E":
                    pygame.draw.rect(
                        display_surf,
                        GREEN,
                        (
                            j * self.tile_size_x,
                            i * self.tile_size_y,
                            self.tile_size_x,
                            self.tile_size_y,
                        ),
                    )

                # for i in range(4):
                #     pygame.draw.rect(
                #         display_surf,
                #         (255, 0, 0),
                #         (
                #             (j * self.tile_size_x) - i,
                #             (j * self.tile_size_y) - i,
                #             self.tile_size_x,
                #             self.tile_size_y,
                #         ),
                #         1,
                #     )

        for item in self.coinList:
            # print("item", item, item.x, item.y, item.width, item.height)
            display_surf.blit(self._coin_surf, item.topleft)
            pygame.draw.rect(
                display_surf,
                (255, 215, 0),
                (
                    item.x,
                    item.y,
                    item.width,
                    item.height,
                ),
            )

        for item in self.treasureList:
            # display_surf.blit(self._treasure_surf, item.topleft)
            pygame.draw.rect(
                display_surf,
                (139, 69, 19),
                (
                    item.x,
                    item.y,
                    item.width,
                    item.height,
                ),
            )

        for item in self.obstacleList:
            pygame.draw.rect(display_surf, RED, item)

        for item in self.monsterList:
            display_surf.blit(self._monster_surf, item.rect.topleft)

    def make_perception_list(self, player_current, display_surf):
        perception_distance = PERCEPTION_RADIUS * max(
            self.tile_size_x, self.tile_size_y
        )
        perception_left = player_current.x + 0.5 * (
            player_current.size_x - perception_distance
        )
        perception_top = player_current.y + 0.5 * (
            player_current.size_y - perception_distance
        )
        perception_rect = pygame.Rect(
            perception_left, perception_top, perception_distance, perception_distance
        )
        wall_list = []
        obstacle_list = []
        item_list = []
        monster_list = []
        for i in perception_rect.collidelistall(self.wallList):
            wall_list.append(self.wallList[i])
        for i in perception_rect.collidelistall(self.obstacleList):
            obstacle_list.append(self.obstacleList[i])
        for i in perception_rect.collidelistall(self.coinList):
            item_list.append(self.coinList[i])
        for i in perception_rect.collidelistall(self.treasureList):
            item_list.append(self.treasureList[i])

        # POUR DEBUG - tenir la touche "p" pour voir la zone de perception
        # pygame.draw.rect(display_surf, GREEN, perception_rect)
        thick = 2
        color_w = (200, 100, 10)
        pygame.draw.line(
            display_surf,
            color_w,
            perception_rect.topleft,
            perception_rect.topright,
            thick,
        )
        pygame.draw.line(
            display_surf,
            color_w,
            perception_rect.topright,
            perception_rect.bottomright,
            thick,
        )
        pygame.draw.line(
            display_surf,
            color_w,
            perception_rect.bottomright,
            perception_rect.bottomleft,
            thick,
        )
        pygame.draw.line(
            display_surf,
            color_w,
            perception_rect.bottomleft,
            perception_rect.topleft,
            thick,
        )
        pygame.display.flip()
        # print([wall_list, obstacle_list, item_list, monster_list])
        return [wall_list, obstacle_list, item_list, monster_list]
