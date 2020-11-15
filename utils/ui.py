import time
import sys


def get_arrow(num):
    s = "="
    for i in range(num):
        if i == 0:
            sys.stdout.write(f'{s}>' + "\n")
        else:
            sys.stdout.write(f'\x1B[1A{s}>' + "\n")
            time.sleep(0.1)
        s += "="


get_arrow(50)
