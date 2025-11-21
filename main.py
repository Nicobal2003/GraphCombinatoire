from read_input import read_input
from initWH import initial_solution


X, m, n, r, LW, UW, LH, UH = read_input('input.txt')
print("X: ", X)
print("m: ", m)
print("n: ", n)
print("r: ", r)
print("LW: ", LW)
print("UW: ", UW)
print("LH: ", LH)
print("UH: ", UH)

W,H = initial_solution(m, n, r, LW, UW, LH, UH)

print ("W : ", W )
print("H : ", H)