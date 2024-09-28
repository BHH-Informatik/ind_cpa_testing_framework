#! /usr/bin/env python3

import math
from Crypto.Util.number import getPrime, inverse

class Algorithm:
    def __init__(self, size):
        self.size = size
    # Key generation
    def generate_keys(self): # low exponent for vulnerability
        bits = self.size
        low_exponents=[3, 5, 7, 11, 17, 257]

        p = getPrime(bits // 2)
        q = getPrime(bits // 2)
        n = p * q
        phi_n = (p - 1) * (q - 1)
        
        # Ensure e is coprime to phi(n)
        for e in low_exponents:
            if math.gcd(e, phi_n) == 1:
                break
        else:
            raise ValueError("e must be coprime with phi(n)")
                
        d = inverse(e, phi_n)
        return (n, e), (n, d)

    # Encryption (no padding)
    def encrypt(self, plaintext, public_key):
        n = public_key[0][0]
        e = public_key[0][1]
        cyphertext = [pow(int(i), e, n) for i in plaintext]
        return cyphertext

    # Decryption
    def decrypt(self, cyphertext, private_key):
        n = private_key[1][0]
        d = private_key[1][1]        
        plaintext = [pow(int(i), d, n) for i in cyphertext]
        return plaintext
