nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [H1|T1]) :- digit(H, H1), digit_list(T, T1).

digit_list_2d([],[]).
digit_list_2d([H|T], [N|TN]) :- digit_list(H, N), digit_list_2d(T, TN).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr), Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

num_list_2d([], []).
num_list_2d([H|T], [N|TN]) :- num_list(H, N), num_list_2d(T, TN).

replace([_|T], 0, E, [E|T]). 
replace([H|T], N, E, [H|TE]) :- not(N is 0), 
                                N1 is N - 1, replace(T, N1, E, TE). 

replace_2d([H|T], (0, C), E, [NH|T]) :- replace(H, C, E, NH).
replace_2d([H|T], (R, C), E, [H|NT]) :- not(R is 0),
                                        R1 is R - 1, replace_2d(T, (R1, C), E, NT).

nth_2d(X, (R, C), E) :- nth0(R, X, L), nth0(C, L, E).

create(0, []).
create(N, [0|T]) :- not(N is 0),
                    N1 is N - 1, create(N1, T).

create_2d(0, _, []).
create_2d(M, N, [H|T]) :- not(M is 0),
                          create(N, H), M1 is M - 1, create_2d(M1, N, T). 

update(M, V, (0, 0), NM) :- nth_2d(V, (0,0), E), replace_2d(M, (0,0), E, NM).
update(M, V, (0, C), NM) :- not(C is 0),
                            C1 is C - 1, 
                            nth_2d(V, (0,C), N1), nth_2d(M, (0, C1), N2),
                            N is N1 + N2,
                            replace_2d(M, (0,C), N, NM).
update(M, V, (R, 0), NM) :- not(R is 0),
                            R1 is R - 1, 
                            nth_2d(V, (R,0), N1), nth_2d(M, (R1, 0), N2),
                            N is N1 + N2,
                            replace_2d(M, (R,0), N, NM).
update(M, V, (R, C), NM) :- not(R is 0), not(C is 0),
                            C1 is C - 1, R1 is R - 1,
                            nth_2d(V, (R,C), N1),
                            nth_2d(M, (R1,C), V1), nth_2d(M, (R, C1), V2),
                            N2 is min(V1, V2),
                            N is N1 + N2,
                            replace_2d(M, (R,C), N, NM).

next_dir(M, (R, C), (R, C1)) :- C1 is C + 1, nth_2d(M, (R, C1), _). 
next_dir(M, (R, C), (R1, 0)) :- C1 is C + 1, not(nth_2d(M, (R, C1), _)),
                                R1 is R + 1, nth_2d(M, (R1, 0), _).

path_sum(M, V, C, NM) :- update(M, V, C, R), next_dir(M, C, NC),
                         path_sum(R, V, NC, NM).
path_sum(M, V, C, NM) :- update(M, V, C, NM), not(next_dir(M, C, _)).     

min_path([], 0).
min_path([H|T], N) :- length([H|T], D1), length(H, D2), create_2d(D1, D2, M),
                      path_sum(M, [H|T], (0,0), NM),
                      C1 is D1 - 1, C2 is D2 - 1,
                      nth_2d(NM, (C1, C2), N).

% Assume: input given as list of lists of lists [[[1], [2], [3]], [[4], [5], [6]]]
minimum_path_sum(X, N) :- digit_list_2d(X, M), min_path(M, N).