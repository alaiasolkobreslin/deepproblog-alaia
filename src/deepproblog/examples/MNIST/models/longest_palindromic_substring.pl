nn(emnist_net,[X],Y,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']) 
                     :: char(X,Y).

char_list([], []).
char_list([H|T], [H1|T1]) :- char(H, H1), char_list(T, T1).

palindrome([]).
palindrome([_]).
palindrome(L) :- append([H|T], [H], L),
                 palindrome(T).

longest_palindrome(L, L) :- palindrome(L).
longest_palindrome([H|T], L2) :- not(palindrome([H|T])),
                                 last(T, T1),
                                 append(HT, [T1], T),
                                 longest_palindrome([H|HT],L1),
                                 longest_palindrome(T,L2),
                                 length(L1, N1), length(L2, N2),
                                 N1 < N2.
longest_palindrome([H|T], L1) :- not(palindrome([H|T])),
                                 last(T, T1),
                                 append(HT, [T1], T),
                                 longest_palindrome([H|HT],L1),
                                 longest_palindrome(T,L2),
                                 length(L1, N1), length(L2, N2),
                                 N1 >= N2.

% Assume: input aba given as [['a'], ['b'], ['a']]    
longest_palindromic_substring(X, S) :- char_list(X, L),
                                      longest_palindrome(L, L1),
                                      string_chars(S, L1).