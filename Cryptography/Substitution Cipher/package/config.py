from typing import Union

import sys
import os

import colored


# Color variables
F = colored.fg("#ffffff") # White
B = colored.fg("#999999") # Gray
R = colored.fg("#cc2023") # Red

# Alphabeth in use
alph = list("ABCDEFGHIJKLMNOPRSTUVWXYZÆØÅ")

# Function to clear the screen, cross platform
clear = lambda : os.system('cls' if os.name == 'nt' else 'clear')


def ask(prompt: str, newLine: bool = True, parseInt: bool = False) -> Union[str,int]:
    """ Prompt the user for an input in a unified way """
    
    brk = "\n" if prompt else "" # If there is no prompt, dont add newline before 
    resp = input(f"{F}{prompt}{brk}{B}>> {R}")

    if parseInt:
        if resp.isdigit():
            return int(resp)
        return ask("Your input has to be a number.", newLine = newLine, parseInt = parseInt)

    print(F, end="\n" if newLine else "", flush = True)
    return resp


def safe_exit() -> None:
    """ Exit the program """
    sys.exit(0)

