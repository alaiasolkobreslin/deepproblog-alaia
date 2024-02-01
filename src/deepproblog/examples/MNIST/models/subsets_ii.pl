nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

number([],Result,Result).
number([H|T],Acc,Result) :- digit(H,Nr),Acc2 is Nr+10*Acc,number(T,Acc2,Result).
number(X,Y) :- number(X,0,Y).

num_list([], []).
num_list([H|T], [N|TN]) :- number(H, N), num_list(T, TN).

digit_list([], []).
digit_list([H|T], [N|TN]) :- digit(H, N), digit_list(T, TN).

subset([], []).
subset([H|T], [H|S]) :- subset(T, S).
subset([_|T], S) :- subset(T, S).

remove_dup([], []).
remove_dup([H|T], Y) :- member(H, T),
                        remove_dup(T, Y).
remove_dup([H|T], [H|Y]) :- not(member(H,T)),
                            remove_dup(T, Y).

% Assume list of lists as input [[1], [2], [3]]
subsets_ii(X, Y) :- digit_list(X, XL),
                    sort(0, @=<, XL, L),
                    findall(T, subset(L, T), S),
                    remove_dup(S, Y).