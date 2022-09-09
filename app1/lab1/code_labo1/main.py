# Université de Sherbrooke
# Code préparé par Audrey Corbeil Therrien
# Laboratoire 1 - Interaction avec prolog

from swiplserver import PrologMQI

print("prolog server started")

if __name__ == "__main__":
    with PrologMQI() as mqi:
        # with mqi.create_thread() as prolog_thread:
        #     result = prolog_thread.query("member(X, [first, second, third]).")
        #     print("member(X, [first, second, third])", result)

        with PrologMQI() as mqi_file:
            with mqi_file.create_thread() as prolog_thread:

                result = prolog_thread.query("[prolog/maze].")
                print("[prolog/maze].", result)

                result = prolog_thread.query("[prolog/functions].")
                print("[prolog/functions].", result)

                # print("prolog_thread", prolog_thread)

                # # Load a prolog file
                # result = prolog_thread.query("[prolog/parente].")
                # print("[prolog/parente].", result)

                # print("\n\n")

                # # # Query the information in the file
                # result = prolog_thread.query("homme(X).")
                # print("homme(X).", result)

                # louis_homme = prolog_thread.query("homme(louis).")
                # print("homme(louis)", louis_homme)

                # maze_1 = prolog_thread.query("maze(1, 1, X).")
                # print("cell(1, 1, X)", maze_1

                # all_empty_cells = prolog_thread.query("maze(X, Y, 0).")
                # print("all_empty_cells", all_empty_cells)

                # maze_1 = prolog_thread.query("maze(X, Y, 0).")
                # print("maze", maze_1)

                all_moves = prolog_thread.query("passage(2, 2, X, Y).")
                print("all_moves", all_moves)

    # # Query the information in the file
    # result = prolog_thread.query("fils(luc, X).")
    # print(result)
