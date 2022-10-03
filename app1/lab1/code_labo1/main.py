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

                result = prolog_thread.query("[prolog/notes].")
                # print("[prolog/maze].", result)

                # # Query the information in the file
                # result = prolog_thread.query("note(paul,X,Y).")
                # print("note.", result)
                query = "mystere(1,[0,1,2,3],R)."
                result = prolog_thread.query(query)
                print(query, result)

    # # Query the information in the file
    # result = prolog_thread.query("fils(luc, X).")
    # print(result)
