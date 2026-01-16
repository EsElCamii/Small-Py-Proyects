#This one just calculates num1 and num2 based on selected operator
def simple_calculator():
    operation_type = input("Enter the operation type: 1.Addition, 2.Subtraction, 3.Multiplication, 4.Division... ")

    num1 = int(input("Enter a number.."))
    num2 = int(input("Enter the second number..."))

    if operation_type == "1":
        print(f"The result is {num1 + num2}")

    elif operation_type == "2":
        print(f"The result is {num1 - num2}")

    elif operation_type == "3":
        print(f"The result is {num1 * num2}")

    elif operation_type == "4":
        print(f"The result is {num1 / num2}")
    
    else:
        print("Invalid operation type, please type 1, 2, 3, or 4")

#This is a calculator that tokenises the equation and identifiying operators and operands returns one number
def advanced_calculator():
    equation = input("Enter the equation... ")

    tokens = []
    buffer = ""

#Tokenising the equation, accounting the spaces and . if there are
    for char in equation:
        if char.isdigit() or char == ".":
            buffer += char
        elif char.isspace():
            continue
        elif char in "+-*/":
            if buffer != "":
                tokens.append(float(buffer))
                buffer = ""
            tokens.append(char)

    if buffer != "":
        tokens.append(float(buffer))
        
#Reducing first the mult and div operators using current number, operator, and next number
    reduced = []
    i = 0
    while i < len(tokens):
        if isinstance(tokens[i], (int, float)):
            current = tokens[i]
            while i + 1 < len(tokens) and tokens[i+1] in ("*", "/"):
                opperator = tokens[i+1]
                right = tokens[i+2]
                if opperator == "*":
                    current = current * right
                else:
                    current = current / right
                i += 2
            reduced.append(current)
            i += 1
        else:
            reduced.append(tokens[i])
            i += 1
    
#Reducing the add and sub operators the same as last
    result = reduced[0]
    i = 1
    while i < len(reduced):
        operator = reduced[i]
        right = reduced[i + 1]
        if operator == "+":
            result = result + right
        elif operator == "-":
            result = result - right
        else:
            raise ValueError(f"Invalid operator: {operator}")
        i += 2

    #finally print
    print(f"The result is {result}")






#Init whole calc, with start menu
while True:
    calculator_type = input("Enter the calculator type: 1.Simple Calculator, 2.Advanced Calculator... ")

    if calculator_type == "1":
        simple_calculator()
    else:
        advanced_calculator()
