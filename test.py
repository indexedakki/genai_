def fibonacci(n):
    a = 0
    b = 1

    for i in range(2, n):
        a, b = b, a+b
        yield b

# Example usage:
for num in fibonacci(8):
    print(num)