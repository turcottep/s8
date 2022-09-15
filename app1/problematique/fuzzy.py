import enum
import time


def get_movement_for_ai_fuzzy(
    path, player_position, perception_list, current_step_index
):

    player_position_center = (player_position[0] + 10, player_position[1] + 10)

    player_position_cell_i_j = (
        player_position[0] // 50,
        player_position[1] // 50,
    )

    main_direction_vector = (
        path[current_step_index + 1][0] - path[current_step_index][0],
        path[current_step_index + 1][1] - path[current_step_index][1],
    )

    if main_direction_vector[0] == 1:
        if player_position_center[0] >= path[current_step_index + 1][0] * 50 + 25:
            current_step_index += 1
    elif main_direction_vector[0] == -1:
        if player_position_center[0] <= path[current_step_index + 1][0] * 50 + 25:
            current_step_index += 1
    elif main_direction_vector[1] == 1:
        if player_position_center[1] >= path[current_step_index + 1][1] * 50 + 25:
            current_step_index += 1
    elif main_direction_vector[1] == -1:
        if player_position_center[1] <= path[current_step_index + 1][1] * 50 + 25:
            current_step_index += 1

    current_step = path[current_step_index]
    current_step_position_center = (
        current_step[0] * 50 + 25,
        current_step[1] * 50 + 25,
    )
    next_step = path[current_step_index + 1]
    next_step_position = (next_step[0] * 50 + 25, next_step[1] * 50 + 25)

    main_direction_vector = (
        next_step[0] - current_step[0],
        next_step[1] - current_step[1],
    )

    print("path", path)
    print("current_step_index", current_step_index)
    print("current_step", current_step)
    print("next_step", next_step)

    # fuzzy code

    [output_left, output_right] = [0, 0]

    [output_left, output_right] = get_output_follow_line(
        player_position_center, current_step_position_center, main_direction_vector
    )

    for obstacle in perception_list[1]:
        position_obstacle = (obstacle[0] + 5, obstacle[1] + 5)

        temp = get_output_dodge_obstacle(
            player_position_center,
            position_obstacle,
            main_direction_vector,
            current_step_position_center,
        )
        print("temp", temp)
        factor = 100
        output_left += factor * temp[0]
        print("**output_left", output_left)
        output_right += factor * temp[1]  # + output_right
        print("**output_right", output_right)

    class Move(enum.Enum):
        north = [1, 0, 0, 0]
        east = [0, 1, 0, 0]
        south = [0, 0, 1, 0]
        west = [0, 0, 0, 1]

    instructions = [0, 0, 0, 0]
    if main_direction_vector == (1, 0):
        instructions = Move.east.value
        if output_left > output_right:
            instructions[0] = 1
        elif output_right > output_left:
            instructions[2] = 1
    elif main_direction_vector == (-1, 0):
        instructions = Move.west.value
        if output_left > output_right:
            instructions[2] = 1
        elif output_right > output_left:
            instructions[0] = 1
    elif main_direction_vector == (0, 1):
        instructions = Move.south.value
        if output_left > output_right:
            instructions[3] = 1
        elif output_right > output_left:
            instructions[1] = 1
    elif main_direction_vector == (0, -1):
        instructions = Move.north.value
        if output_left > output_right:
            instructions[1] = 1
        elif output_right > output_left:
            instructions[3] = 1

    print("instructions", instructions)

    # if len(perception_list[1]) > 0:
    #     time.sleep(0.00001)

    # get item, override instructions
    item_list = perception_list[2]
    if len(item_list) > 0:
        item = item_list[0]

        instructions = [0, 0, 0, 0]  # up, right, down, left
        if item[0] > player_position_center[0]:
            instructions[1] = 1
            print("item right")
        elif item[0] < player_position_center[0]:
            instructions[3] = 1
            print("item left")
        if item[1] > player_position_center[1]:
            instructions[2] = 1
            print("item down")
        elif item[1] < player_position_center[1]:
            instructions[0] = 1
            print("item up")

    return [instructions, current_step_index]


def get_output_follow_line(
    player_position_center, line_position, main_direction_vector
):
    if main_direction_vector[0] == 0:
        # vertical line
        distance_line_secondary_direction = (
            line_position[0] - player_position_center[0]
        ) * main_direction_vector[1]
    elif main_direction_vector[1] == 0:
        # horizontal line
        distance_line_secondary_direction = (
            line_position[1] - player_position_center[1]
        ) * main_direction_vector[0]

    if distance_line_secondary_direction > 0:
        return [(distance_line_secondary_direction - 50) / 50, 0]
    elif distance_line_secondary_direction < 0:
        return [0, (distance_line_secondary_direction - 50) / 50]
    else:
        return [0, 0]


def get_output_dodge_obstacle(
    player_position_center, obstacle_position, main_direction_vector, line_position
):

    if main_direction_vector[0] == 0:
        # vertical line
        distance_obstacle = (
            player_position_center[0] - obstacle_position[0]
        ) * main_direction_vector[1]
        temp_for_math = (
            player_position_center[1] - obstacle_position[1]
        ) * main_direction_vector[1]
        if main_direction_vector[1] == 1:
            if temp_for_math < 0:
                scale = 1  # abs(1 / temp_for_math)
            else:
                scale = 0
        else:
            if temp_for_math > 0:
                scale = 0  # abs(1 / temp_for_math)
            else:
                scale = 1

    elif main_direction_vector[1] == 0:
        # horizontal line
        distance_obstacle = (
            player_position_center[1] - obstacle_position[1]
        ) * main_direction_vector[0]
        temp_for_math = (
            player_position_center[0] - obstacle_position[0]
        ) * main_direction_vector[0]
        if main_direction_vector[0] == 1:
            if temp_for_math < 0:
                scale = 1  # abs(1 / temp_for_math)
            else:
                scale = 0
        else:
            if temp_for_math > 0:
                scale = 0  # abs(1 / temp_for_math)
            else:
                scale = 1
    print("player_position_center", player_position_center[1])
    print("obstacle_position", obstacle_position[1])
    print("distance_obstacle", distance_obstacle)

    if distance_obstacle >= 0 and distance_obstacle < 16:
        output_left = (16 - distance_obstacle) / 32
        output_right = (16 - distance_obstacle) / 16
        print("case 1")
    elif distance_obstacle < 0 and distance_obstacle > -16:
        output_left = abs((16 + distance_obstacle)) / 16
        output_right = abs((16 + distance_obstacle)) / 32
        print("case 2")
    else:
        output_left = 0
        output_right = 0
        print("case 3")

    temp_r4 = r4(line_position, obstacle_position, main_direction_vector)
    # scale_factor = math.sqrt(((obstacle_position[0] + player_position_center[0]) ** 2) + (
    #    (player_position_center[1] + obstacle_position[1]) ** 2
    # )) * 200
    # print("________", scale_factor)

    print("R3", output_left, " ", output_right)
    print("R4", temp_r4)

    output_left = min(output_left, temp_r4[0]) * scale
    output_right = min(output_right, temp_r4[1]) * scale

    return [output_left, output_right]


def r4(line_position, obstacle_position, main_direction_vector):

    if main_direction_vector[0] == 0:
        # vertical line
        distance_obstacle_center = (
            line_position[0] - obstacle_position[0]
        ) * main_direction_vector[1]
    elif main_direction_vector[1] == 0:
        # horizontal line
        distance_obstacle_center = (
            line_position[1] - obstacle_position[1]
        ) * main_direction_vector[0]

    if distance_obstacle_center < 0:
        output_left = abs(distance_obstacle_center) / 25
        output_right = 0
    elif distance_obstacle_center > 0:
        output_left = 0
        output_right = abs(distance_obstacle_center) / 25
    else:
        output_left = 1
        output_right = 1

    return [output_left, output_right]
