nn(mnist_net,[X],Y,[0,1,2,3,4,5,6,7,8,9]) :: digit(X,Y).

digit_list([], []).
digit_list([H|T], [H1|T1]) :- digit(H, H1), digit_list(T, T1).

add_digits(X, Y, Z, 1, N) :- X + Y + Z >= 10, 
                             N is (X+Y+Z) mod 10.
add_digits(X, Y, Z, 0, N) :- X + Y + Z < 10,
                          N is X + Y + Z.

add_with_carry([], [], 0, []).
add_with_carry([], [], 1, [1]).
add_with_carry([H|T],[],C,[N|T2]) :- add_digits(H,0,C,NC,N),
                                     add_with_carry(T,[],NC,T2).
add_with_carry([],[H|T],C,[N|T2]) :- add_digits(0,H,C,NC,N),
                                     add_with_carry([],T,NC,T2).
add_with_carry([H1|T1],[H2|T2],C,[N|NT]) :- add_digits(H1,H2,C,NC,N),
                                            add_with_carry(T1,T2,NC,NT).

% Assume [[2],[4],[3]], [[5],[6],[4]]               
add_two_numbers(X,Y,N) :- digit_list(X,X2),
                          digit_list(Y,Y2),
                          add_with_carry(X2,Y2,0,N).