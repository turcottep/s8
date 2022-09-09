libre(X,Y) :- maze(X,Y,0).

passage(X1, Y1, X2, Y1) :- X2 is X1+1, libre(X2,Y1).
passage(X1, Y1, X2, Y1) :- X2 is X1-1, libre(X2,Y1).
passage(X1, Y1, X1, Y2) :- Y2 is Y1+1, libre(X1,Y2).
passage(X1, Y1, X1, Y2) :- Y2 is Y1-1, libre(X1,Y2).
