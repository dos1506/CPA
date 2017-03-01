import json

with open('randomText') as f:
    data = f.readlines()

hexPlainTexts = data

plainTexts = [' '.join([str(int(hexPlainTexts[i][x*2:x*2+2], 16)) for x in range(int(len(hexPlainTexts[i])/2))]) for i in range(len(hexPlainTexts))]

#cipherTexts = [[int(hexCipherTexts[i][x*2:x*2+2], 16) for x in range(int(len(hexCipherTexts[i])/2))] for i in range(len(hexCipherTexts))]

with open('plain.txt', 'w+') as f:
    f.write('\n'.join(plainTexts))

#with open('cipher.txt', 'w+') as f:
#    f.write('\n'.join([' '.join(map(str, x)) for x in cipherTexts]))
