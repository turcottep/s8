from pygame.locals import *
import pygame

from Player import *
from Maze import *
from Constants import *
from astar import astar


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
        self.maze = Maze(mazefile)

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

        # Utility functions for AI
        if keys[K_p]:
            print("perception:")
            print(self.maze.make_perception_list(self.player, self._display_surf))
            print("")
            # returns a list of 4 lists of pygame.rect inside the perception radius
            # the 4 lists are [wall_list, obstacle_list, item_list, monster_list]
            # item_list includes coins and treasure

        if keys[K_m]:
            for monster in self.maze.monsterList:
                print(monster.mock_fight(self.player))
            # returns the number of rounds you win against the monster
            # you need to win all four rounds to beat it

        if keys[K_ESCAPE]:
            self._running = False

    # FONCTION À Ajuster selon votre format d'instruction
    def on_AI_input(self, instruction):
        # print("instruction", instruction)
        # "instruction" est une liste de 4 éléments [up, right, down, left]

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

    def on_render(self):
        self.maze_render()
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

        path = do_planification()
        do_genetics()

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
            check_for_incoming_monster_to_set_stats()
            key_to_press_for_AI = get_movement_for_ai(path, self.player.get_position())
            self.on_AI_input(key_to_press_for_AI)
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
            self.on_render()

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


def get_movement_for_ai(path, player_position):
    player_position_cell_i_j = (player_position[0] // 50, player_position[1] // 50)
    current_step = path.index(player_position_cell_i_j)
    print("current_step", current_step)
    next_step = path[current_step + 1]
    next_step_position = (next_step[0] * 50 + 25, next_step[1] * 50 + 25)

    instructions = [0, 0, 0, 0]  # up, right, down, left

    if next_step_position[0] > player_position[0] + 5:
        instructions[1] = 1
    elif next_step_position[0] < player_position[0] + 5:
        instructions[3] = 1
    if next_step_position[1] > player_position[1] + 5:
        instructions[2] = 1
    elif next_step_position[1] < player_position[1] + 5:
        instructions[0] = 1

    return instructions


def do_planification():

    # planif = [
    #     (1, 0),
    #     (1, 1),
    #     (1, 2),
    #     (1, 3),
    #     (1, 4),
    #     (1, 5),
    #     (1, 6),
    #     (1, 7),
    #     (2, 7),
    #     (3, 7),
    #     (4, 7),
    #     (4, 8),
    #     (4, 9),
    #     (5, 9),
    #     (6, 9),
    #     (7, 9),
    #     (8, 9),
    #     (9, 9),
    #     (9, 10),
    #     (9, 11),
    #     (10, 11),
    #     (11, 11),
    #     (12, 11),
    #     (13, 11),
    #     (13, 12),
    #     (13, 13),
    #     (14, 13),
    #     (15, 13),
    #     (16, 13),
    #     (17, 13),
    #     (17, 14),
    #     (18, 14),
    #     (19, 14),
    #     (20, 14),
    #     (21, 14),
    #     (22, 14),
    #     (22, 15),
    # ]

    # A-star
    starting_positon = (1, 0)
    ending_position = (22, 15)

    path = astar(starting_positon, ending_position)
    print("path", path)
    return path


def do_genetics():
    pass


def check_for_incoming_monster_to_set_stats():
    pass
