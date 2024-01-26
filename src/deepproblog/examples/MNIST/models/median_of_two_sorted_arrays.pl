nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

merge_sorted([], Y, Y).
merge_sorted(X, [], X).
merge_sorted([HX|TX], [HY|TY], [HX|T]) :- HX < HY, !, 
                                          merge_sorted(TX, [HY|TY], T).
merge_sorted([HX|TX], [HY|TY], [HX,HY|T]) :- HX is HY, !,
                                              merge_sorted(TX, TY, T).
merge_sorted([HX|TX], [HY|TY], [HY|T]) :- merge_sorted([HX|TX], TY, T).                                             

get_median([N], N).
get_median([A,B], N) :- N is A + B / 2.
get_median(L, N) :- append([_|T1], [_], L),
                    get_median(T1, N).

% Assume list of lists as input [[1, 3], [2, 4]]
get_median_two_sorted(X, Y, N) :- num_list(X, L1),
                                  num_list(Y, L2),
                                  merge_sorted(L1, L2, L),
                                  get_median(L, N).