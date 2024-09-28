from Crypto.Cipher import DES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

def encrypt_des(plaintext, key):
    cipher = DES.new(key, DES.MODE_ECB)
    padded_plaintext = pad(plaintext, DES.block_size)
    ciphertext = cipher.encrypt(padded_plaintext)
    return ciphertext

def decrypt_des(ciphertext, key):
    cipher = DES.new(key, DES.MODE_ECB)
    decrypted_padded_plaintext = cipher.decrypt(ciphertext)
    plaintext = unpad(decrypted_padded_plaintext, DES.block_size)
    return plaintext

# Example usage:
# Make sure to replace 'your_key_here' and 'your_plaintext_here' with your actual key and plaintext
key = b'pwhjkezs'  # 8 bytes for DES
plaintext = b'12dqwdwad awd awdawdawdawdwa'

ciphertext = encrypt_des(plaintext, key)
print("Encrypted ciphertext:", ciphertext)

plaintext = decrypt_des(ciphertext, key)
print("Decrypted plaintext:", plaintext.decode('utf-8'))
