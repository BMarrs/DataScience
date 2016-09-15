import sys

input_list = [1,2,3,4,5,6,7,8,27,12]

input_list = []
for line in sys.stdin:
    new_list = [int(elem) for elem in line.split()]
    input_list.append(new_list)

def factor_finder(in_num):
    factor_list = set()
    for i in range(1, int(in_num ** 0.5) + 1):
        d, m = divmod(in_num, i)
        if m == 0:
            factor_list.add(d)
    return len(factor_list)


def sum_factor(input_list):
    """retrieve the unique values from the input string"""
    counter = []
    for ele in input_list:
        ele_val = int(ele)
        factor_cnt = factor_finder(ele_val)
        if factor_cnt == 3:
            counter.append(ele_val)
    sum_val = 0
    for elec in counter:
        sum_val += elec
    print(sum_val)

sum_factor(input_list)
