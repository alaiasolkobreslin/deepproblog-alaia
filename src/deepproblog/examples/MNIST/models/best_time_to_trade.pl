nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

profit([H1, H2|T], N) :- max_list([H2|T], M),
                         H1 =< M,
                         N is M - H1.
profit(_, 0).

max_profit([], 0).
max_profit([_], 0).
max_profit([H|T], N) :- profit([H|T], N1),
                        max_profit(T, N2),
                        N is max(N1, N2).

% Assume list of lists as input [[1], [2], [5]]
best_time_to_trade(L, N) :- num_list(L, L2),
                            max_profit(L2, N).