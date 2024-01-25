nn(emnist_net,[X],Y,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']) 
                     :: char(X,Y).

% Base case: Reversing an empty list results in an empty list
reverse_list([], []).

% Recursive case: Reverse the tail of the list and append the head at the end
reverse_list([Head|Tail], Reversed) :-
    reverse_list(Tail, ReversedTail),
    append(ReversedTail, [Head], Reversed).

reverse_string(A, B, C, Z) :- char(A,A2), char(B,B2), char(C,C2),
                              reverse_list([A2, B2, C2], Z).