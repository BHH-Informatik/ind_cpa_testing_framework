import sys
import AlgorithmusTesterMT
import CustomNetworkLayouts
import string
import random

### Globals

#Global-Data
MODELNAME = "caesar"
SAMPLES = 2**12
DESCRIPTION = "No_Config,String 8 lowercase"

#Train Variables
BATCH_SIZE = 32
EPOCHS = 100
Length = 8

def format(input):
    return bytearray(input).decode("utf-8")
###

def randomString():
    random.seed("123")
    stringLength = Length
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def encrypt(input):
    SHIFT = 10
    result = ""
    # transverse the plain text
    for i in range(len(input)):
        char = input[i]

        # Encrypt uppercase characters in plain text
        if (char.isupper()):
            result += chr((ord(char) + SHIFT - 65) % 26 + 65)
        # Encrypt lowercase characters in plain text
        else:
            result += chr((ord(char) + SHIFT - 97) % 26 + 97)
    return result


if __name__ == '__main__':

    tester = AlgorithmusTesterMT.AlgorithmusTesterMT(MODELNAME, SAMPLES, BATCH_SIZE, EPOCHS, DESCRIPTION, network=CustomNetworkLayouts.defaultHiddenLayers, maxIntValue=Length, networkDataType=AlgorithmusTesterMT.NetworkDataType.BIT)
    if sys.argv[1] == 'generate-training-file':
        print('Generating training file...')
        tester.generate_training_data(randomString, encrypt)
    elif sys.argv[1] == 'train':
        print('Training...')
        tester.train()
    elif sys.argv[1] == 'evaluate':
        print('Evaluate...')
        tester.evaluate(format_cipher=format, format_plain=format)

    print('Done!')
