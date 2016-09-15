import sys

def Staircase(n):
    i = 1
    while i <= n:
        space_count = n - i
        hash_count = i
        print(' ' * (space_count) + '#' * hash_count)
        i += 1

Staircase(6)

