from builtins import print


def A_divided_by_B(A , B):
    print("{A} : {B}".format(A=A, B=B))
    if (B == 0):
        print("No answer")

    a = abs(A)
    b = abs(B)
    count = 0

    while(a >= b):
        a = a - b
        count += 1

    if ((A < 0) ^ (B < 0)):
        count = -count

    print("The result : %2d"%count)


A_divided_by_B(0,-10)