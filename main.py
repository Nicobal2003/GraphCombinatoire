import time
from read_input import read_input

start_time = time.perf_counter()

X, m, n, r, LW, UW, LH, UH = read_input('input.txt')
print("X: ", X)
print("m: ", m)
print("n: ", n)
print("r: ", r)
print("LW: ", LW)
print("UW: ", UW)
print("LH: ", LH)
print("UH: ", UH)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"\nExecution time: {execution_time:.6f} seconds")