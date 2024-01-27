nn(emnist_net,[X],Y,['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 
                     'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                     'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                     'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']) 
                     :: char(X,Y).

% Mapping digits to 0-9
char_value('0', 0).  char_value('1', 1).  char_value('2', 2).  char_value('3', 3).
char_value('4', 4).  char_value('5', 5).  char_value('6', 6).  char_value('7', 7).
char_value('8', 8).  char_value('9', 9).

% Mapping uppercase letters to 10-35
char_value('A', 10). char_value('B', 11). char_value('C', 12). char_value('D', 13).
char_value('E', 14). char_value('F', 15). char_value('G', 16). char_value('H', 17).
char_value('I', 18). char_value('J', 19). char_value('K', 20). char_value('L', 21).
char_value('M', 22). char_value('N', 23). char_value('O', 24). char_value('P', 25).
char_value('Q', 26). char_value('R', 27). char_value('S', 28). char_value('T', 29).
char_value('U', 30). char_value('V', 31). char_value('W', 32). char_value('X', 33).
char_value('Y', 34). char_value('Z', 35).

% Mapping selected lowercase letters to 36-47
char_value('a', 36). char_value('b', 37). char_value('d', 38). char_value('e', 39).
char_value('f', 40). char_value('g', 41). char_value('h', 42). char_value('n', 43).
char_value('q', 44). char_value('r', 45). char_value('t', 46).



% Base case: Reversing an empty list results in an empty list
reverse_list([], []).

% Recursive case: Reverse the tail of the list and append the head at the end
reverse_list([Head|Tail], Reversed) :-
    reverse_list(Tail, ReversedTail),
    append(ReversedTail, [Head], Reversed).




reverse_string(A, B, C, Z) :- char(A,A2), char(B,B2), char(C,C2),
                              reverse_list([A2, B2, C2], Z).