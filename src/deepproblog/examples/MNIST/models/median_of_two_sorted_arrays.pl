nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- digit(H,N), num_list(T, TN).

% Base case: an empty list is already sorted
merge_sort([], []).

% Base case: a list with one element is already sorted
merge_sort([X], [X]).

% Split a list into two halves
split(List, Left, Right) :-
    length(List, Len),
    HalfLen is Len // 2,
    length(Left, HalfLen),
    append(Left, Right, List).

% Merge two sorted lists into a single sorted list
merge([], Right, Right).
merge(Left, [], Left).
merge([X | LeftRest], [Y | RightRest], [X | MergedRest]) :-
    (Y >= X),
    merge(LeftRest, [Y | RightRest], MergedRest).
merge([X | LeftRest], [Y | RightRest], [Y | MergedRest]) :-
    X > Y,
    merge([X | LeftRest], RightRest, MergedRest).

% Recursive case: merge sort
merge_sort(List, Sorted) :-
    split(List, Left, Right),
    merge_sort(Left, SortedLeft),
    merge_sort(Right, SortedRight),
    merge(SortedLeft, SortedRight, Sorted).

merge_sorted([], Y, Y).
merge_sorted(X, [], X).
merge_sorted([HX|TX], [HY|TY], [HX|T]) :- HX < HY,
                                          merge_sorted(TX, [HY|TY], T).
merge_sorted([HX|TX], [HY|TY], [HX,HY|T]) :- HX is HY,
                                             merge_sorted(TX, TY, T).
merge_sorted([HX|TX], [HY|TY], [HY|T]) :- merge_sorted([HX|TX], TY, T).                                             

get_median([N], N).
get_median([A,B], N) :- N is A + B / 2.
get_median(L, N) :- append([_|T1], [_], L),
                    get_median(T1, N).

% Assume list of lists as input [[1, 3], [2, 4]]
get_median_two_sorted(A, B, C, D, E, F, N) :- num_list([A,B,C], L1),
                                              num_list([D,E,F], L2),
                                              merge_sort(L1, S1),
                                              merge_sort(L2, S2),
                                              merge_sorted(S1, S2, L),
                                              get_median(L, N).