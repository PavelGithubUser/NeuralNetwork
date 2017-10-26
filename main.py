import os
from src import test
from src import train


def cls():
    os.system('cls' if os.name=='nt' else 'clear')

cls()

print("Viberite variant: ")
print("1 training'")
print("2 test")

flag = input()

if (flag== "1"):
    train.launch()
elif (flag== "2"):
    test.launch()