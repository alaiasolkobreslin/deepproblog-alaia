nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

reachable([_]).
reachable([H|T]):- length(T, N), H >= N.
reachable([H|T]):- append([H1|T1], [H2|T2], T),
                   reachable([H, H1|T1]),
                   last([H1|T1], TN),
                   reachable([TN, H2|T2]).

% Assume list of lists as input [[3], [0], [1]]
% return 1 for true, 0 for false
jump_game(X, Z) :- num_list(X, L),
                   reachable(L), !, Z is 1.
jump_game(_, Z) :- Z is 0.