

import random

class Key:
    """ Create a key from small passphrase """
    def __init__(self, passwd:str = "") -> None:
        self.raw = list("ABCDEFGHIJKLMNOPRSTUVWXYZÆØÅ")
        self.key = list("ABCDEFGHIJKLMNOPRSTUVWXYZÆØÅ")
        
        random.seed(passwd) # Set the seed with random so we always get the same key from the passphrase

        for i in range(5):
            random.shuffle(self.key)


class Crypter:
    """ Handle the cryptographic part of algarythem """
    def __init__(self, key: Key) -> None:
        self.myKey = key

    def swap_letter(self, letter: str, arr1: list, arr2: list) -> str:
        """ Swaps a letter with another """
        try:
            indx = arr1.index(letter.upper())
        except ValueError:
            return letter 

        return arr2[indx]

    def encrypt(self, phrase:str) -> str:
        """ Encrypt a message using the key """
        txt = "".join([self.swap_letter(i, self.myKey.key, self.myKey.raw) for i in phrase])
        return txt

    def decrypt(self, phrase:str) -> str:
        """ Decrypt a message using the key """
        txt = "".join([self.swap_letter(i, self.myKey.raw, self.myKey.key) for i in phrase])
        return txt



if __name__ == "__main__":
    # Create the key object
    myKey = Key("passwdei23")
    print(myKey.key)
    # Create the ecnryption obejct, it takes a key and 
    crypt = Ceasar(myKey)

    with open("chiffer.txt", "r") as file:
        chiffer = crypt.encrypt(file.read())

    with open("chiffer.txt", "w") as file:
        file.write(chiffer)
