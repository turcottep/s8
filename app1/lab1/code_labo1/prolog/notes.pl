note(paul,bio,51).
note(bob,chimie,24).

estChaud(cuisine).
estChaud(bois).

enFumee(bois).
enFumee(table).

enFeu(X) :- estChaud(X), enFumee(X).


mystere( _ , [] , [] ).
mystere(X,[X|T],R) :-mystere(X,T,R).
mystere( X, [Y | T], [Y | R] ) :- X \= Y, mystere( X, T, R ).
