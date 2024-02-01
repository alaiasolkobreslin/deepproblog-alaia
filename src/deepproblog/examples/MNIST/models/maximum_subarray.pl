nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [N|TN]) :- digit(H, N), digit_list(T, TN).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

sum([], 0).
sum([H|T], N) :- sum(T, M),
                 N is H + M.

max_subarray([], 0).
max_subarray([H], H).
max_subarray([H1,H2|T], N):- sum([H1,H2|T], N1),
                             last([H2|T], T1),
                             append(HT, [T1], [H2|T]),
                             max_subarray([H1|HT], N2),
                             max_subarray([H2|T], N3),
                             M1 is max(N2, N3),
                             N is max(M1, N1).

% Assume list of lists as input [[1], [5], [4]]
maximum_subarray(X, N) :- digit_list(X, L), 
                          max_subarray(L, N).