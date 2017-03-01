from Crypto import Random
from Crypto.Cipher import AES
from tqdm import tqdm

plainTextFile = 'randomText_200000'

with open(plainTextFile) as f:
    plainText = f.read().splitlines()

key = bytes.fromhex('2b7e151628aed2a6abf7158809cf4f3c')
cipher = AES.new(key, AES.MODE_ECB)

cipherText = list()
for i in tqdm(range(len(plainText))):
    c = cipher.encrypt(bytes.fromhex(plainText[i]))
    cipherText.append(' '.join([str(int(b)) for b in c]))

cipherTextFile = 'cipherText_200000'

with open(cipherTextFile, 'w+') as f:
    f.write('\n'.join(cipherText))

