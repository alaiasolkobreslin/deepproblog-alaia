nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [N|TN]) :- digit(H,N), digit_list(T, TN).

merge_sorted([], Y, Y).
merge_sorted(X, [], X).
merge_sorted([HX|TX], [HY|TY], [HX|T]) :- HX < HY,
                                          merge_sorted(TX, [HY|TY], T).
merge_sorted([HX|TX], [HY|TY], [HX,HY|T]) :- HX is HY,
                                             merge_sorted(TX, TY, T).
merge_sorted([HX|TX], [HY|TY], [HY|T]) :- HX > HY, 
                                          merge_sorted([HX|TX], TY, T).                                             

get_median([N], N).
get_median([A,B], N) :- N is (A + B) / 2.
get_median([A,B,C|T], N) :- append([_|T1], [_], [A,B,C|T]),
                            get_median(T1, N).

get_median_two_sorted(A, B, C, D, E, F, N) :- digit_list([A,B,C], L1),
                                              digit_list([D,E,F], L2),
                                              sort(0, @=<, L1, S1),
                                              sort(0, @=<, L2, S2),
                                              merge_sorted(S1, S2, L),
                                              get_median(L, N).