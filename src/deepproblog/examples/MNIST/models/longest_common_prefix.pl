nn(emnist_net,[X],Y,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']) 
                     :: char(X,Y).

char_list([], []).
char_list([H|T], [H1|T1]) :- char(H, H1), char_list(T, T1).

char_list_list([], []).
char_list_list([H|T], [H1|T1]) :- char_list(H, H1), char_list_list(T, T1).

common_prefix([], _, []).
common_prefix([[C|L1]|T], C, [L1|R]) :- common_prefix(T, C, R).

longest_prefix([[H1|T1]|T], S) :- common_prefix([[H1|T1]|T], H1, R), !,
                                  longest_prefix(R, S1),
                                  string_concat(H1, S1, S).
longest_prefix(_, "").

% Assume: input list of lists [[a, b], [a], [a, c]]                               
longest_common_prefix(X, S) :- char_list_list(X, L),
                               longest_prefix(L, S).