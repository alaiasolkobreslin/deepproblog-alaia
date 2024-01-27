nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [H1|T1]) :- digit(H, H1), digit_list(T, T1).

decode([], 1).
decode([0|_], 0) :- !.
decode([_], 1).
decode([H, H1|T], N) :- H is 1, !,
                        decode([H1|T], N1),
                        decode(T, N2),
                        N is N1 + N2.
decode([H, H1|T], N) :- H is 2, H1 =< 6, !,
                        decode([H1|T], N1),
                        decode(T, N2),
                        N is N1 + N2.
decode([_|T], N) :- decode(T, N).

% Assume input 123 given as [1, 2, 3]
decode_ways(X, N) :- digit_list(X, L),
                     decode(L, N).
