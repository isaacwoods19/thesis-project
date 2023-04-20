test_dict = {}

test_dict['a'] = [1,(2,3)]
test_dict['b'] = [4,(5,6)]
test_dict['c'] = [1,(2,3)]

test_dict['c'][0] = 7
test_dict['c'][1] = (8,9)

for letter, [num1, (num2, num3)] in test_dict.items():
    print(letter, num1, num2, num3)