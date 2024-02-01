nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

replace([_|T], 0, E, [E|T]).
replace([H|T], N, E, [H|TE]) :- not(N is 0),
                                N1 is N - 1, replace(T, N1, E, TE). 

replace_2d([H|T], (0, C), E, [NH|T]) :- replace(H, C, E, NH).
replace_2d([H|T], (R, C), E, [H|NT]) :- not(R is 0),
                                        R1 is R - 1, 
                                        replace_2d(T, (R1, C), E, NT).

nth_2d(X, (R, C), E) :- nth0(R, X, L), nth0(C, L, E).

create(0, []).
create(N, [0|T]) :- not(N is 0),
                    N1 is N - 1, create(N1, T).

create_2d(0, _, []).
create_2d(M, N, [H|T]) :- not(M is 0),
                          create(N, H), M1 is M - 1, create_2d(M1, N, T). 

ops(right, (R, C), (R, C1), (R1, C), right, down) :- C1 is C + 1, R1 is R + 1.
ops(left, (R, C), (R, C1), (R1, C), left, up) :- C1 is C - 1, R1 is R - 1.
ops(up, (R, C), (R1, C), (R, C1), up, right) :- C1 is C + 1, R1 is R - 1.
ops(down, (R, C), (R1, C), (R, C1), down, left) :- C1 is C - 1, R1 is R + 1.

next_dir(M, D, C, C1, D1) :- ops(D, C, C1, _, D1, _),
                             nth_2d(M, C1, 0).
next_dir(M, D, C, C2, D2) :- ops(D, C, C1, C2, _, D2),
                             not(nth_2d(M, C1, 0)).

spiral(M, _, _, [], M).
spiral(M, _, C, [H], NM) :- replace_2d(M, C, H, NM).
spiral(M, D, C, [H1,H2|T], R) :- replace_2d(M, C, H1, NM), next_dir(M, D, C, NC, ND),
                                 spiral(NM, ND, NC, [H2|T], R).

spiral_matrix_ii(X, M) :- digit(X, N),
                          N2 is N * N, numlist(1, N2, L), create_2d(N, N, EM),
                          spiral(EM, right, (0,0), L, M).