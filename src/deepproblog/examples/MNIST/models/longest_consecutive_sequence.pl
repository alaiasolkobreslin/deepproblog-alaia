nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [H1|T1]) :- digit(H, H1), digit_list(T, T1).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

consecutive_from([], 0).
consecutive_from([_], 1).
consecutive_from([H1,H2|T], N) :- H2 is H1 + 1,
                                  consecutive_from([H2|T], N1),
                                  N is N1 + 1.
consecutive_from([H1,H2|_], 1) :- not(H2 is H1 + 1).

longest_consecutive([], 0).
longest_consecutive([_], 1).
longest_consecutive([H1,H2|T], N) :- consecutive_from([H1,H2|T], N1),
                                     longest_consecutive([H2|T], N2),
                                     N is max(N1, N2).

% Assume: list of lists as input [[1], [2], [4], [3]]
longest_consecutive_sequence(X, N) :- digit_list(X, L),
                                      sort(L, L1),
                                      longest_consecutive(L1, N).