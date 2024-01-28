nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [N|TN]) :- digit(H, N), digit_list(T, TN).

palindrome([]).
palindrome([_]).
palindrome(L) :- append([H|T], [H], L),
                 palindrome(T).

% Assume: input 1234 given as [1,2,3,4] 
% return 1 for true, 0 for false
main(X, Z) :- digit_list(X, L), 
                           palindrome(L),
                           Z is 1.
main(_, Z) :- Z is 0.

palindrome_number(W, X, Y, Z) :- main([W, X, Y], Z).