nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [H1|T1]) :- digit(H, H1), digit_list(T, T1).

decode([], Z) :- Z is 1.
decode([0|_], Z) :- Z is 0.
decode([H], Z) :- H \= 0, Z is 1. 
decode([1, H1|T], N) :- decode([H1|T], N1),
                        decode(T, N2),
                        N is N1 + N2.
decode([2, H1|T], N) :- H1 =< 6,
                        decode([H1|T], N1),
                        decode(T, N2),
                        N is N1 + N2.
decode([2, H1|T], N) :- H1 > 6,
                        decode([H1|T], N1),
                        decode(T, N2),
                        N is N1 + N2.
decode([H, H1|T], N) :- H > 2, decode([H1|T], N).

decode_ways(A,B,C, N) :- digit_list([A,B,C], L),
                         decode(L, N).
