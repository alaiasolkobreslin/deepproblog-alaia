nn(emnist_net,[X],Y,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']) 
                     :: char(X,Y).

char_list([], []).
char_list([H|T], [H1|T1]) :- char(H, H1), char_list(T, T1).

no_repeat([]).
no_repeat([H|T]) :- not(member(H, T)), no_repeat(T).

longest_no_repeat(L, L, N) :- no_repeat(L), !,
                              length(L, N).
longest_no_repeat([H|T], L2, N2) :- last(T, T1),
                                   append(HT, [T1], T),
                                   longest_no_repeat([H|HT],_,N1),
                                   longest_no_repeat(T,L2,N2),
                                   N1 < N2, !.
longest_no_repeat([H|T], L1, N1) :- last(T, T1),
                                   append(HT, [T1], T),
                                   longest_no_repeat([H|HT],L1,N1).

% Assume: input abc given as list [a, b, c]
longest_substring_no_repeat(X, N) :- char_list(X, L),
                                     longest_no_repeat(L, _, N).