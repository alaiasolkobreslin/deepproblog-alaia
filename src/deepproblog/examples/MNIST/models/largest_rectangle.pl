nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

take_min([H1, H2|T], [H1|L]) :- H1 < H2, !, 
                                take_min([H2|T], L).
take_min([_, H2|T], [H2|L]) :- !, take_min([H2|T], L).
take_min(_, []).

max_area([], _, 0).
max_area([X], N, Y) :- Y is N*X, !.
max_area(X, N, Y) :- max_list(X, A1),
                     Y1 is A1 * N,
                     take_min(X, X2),
                     N2 is N + 1,
                     max_area(X2, N2, Y2),
                     Y is max(Y1, Y2).

% assume list of lists as input [[1], [2], [3], [4]]
largest_rectangle(X, Y) :- num_list(X, L),
                           max_area(L, 1, Y).