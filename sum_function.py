import sys
import re

input_raw = sys.stdin.readline()
input_string = input_raw.split(',')


def string_counter(str_list):
    """retrieve the unique values from the input string"""
    counter = set()
    for ele in str_list:
        if re.match("^[a-zA-Z]*$", ele):
            format_case = ele.lower()
        else:
            format_case = ele
        counter.add(format_case)
    print(len(counter))

string_counter(input_string)
