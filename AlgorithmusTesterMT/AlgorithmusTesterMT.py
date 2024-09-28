import math
import sys
from enum import Enum

from keras.models import Model
from keras import layers, callbacks, losses
import numpy as np
import CustomLosses
import CustomNetworkLayouts
import csv

from time import time
from keras.models import load_model
import os

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class NetworkDataType(Enum):
    """
    Enum to defined the two datatypes
    """
    INT = 1
    BIT = 2


### Helper Functions
def format_nothing(value):
    """
    Default format-function. Doesnt format anything.
    :param value: data to format
    :return: unformated value
    """
    return value


def encodeToNumpyBitArray(input, bits):
    """
    Encode input value to Numpy-Bit-Array of given size.
    :param input: Input-Value
    :param bits: Bit-Size
    :return: umpy-Bit-Array
    """
    if isinstance(input, list):
        return input

    if isinstance(input, (bytes, bytearray)):
        return np.unpackbits(input).astype(dtype=np.float)

    if isinstance(input, int):
        ibytes = bytearray((input).to_bytes(int(math.ceil(bits / 8)), 'big'))
        return np.unpackbits(ibytes).astype(dtype=np.float)[int(math.ceil(bits / 8)) * 8 - bits:]

    mbytes = bytearray(input, 'utf-8')
    return np.unpackbits(mbytes).astype(dtype=np.float)


def encodeToNumpyIntArray(input, maxIntValue):
    """
    Encode input value to scaled Numpy-Int-Array
    :param input: Input-Value
    :param maxIntValue: Max-Value to scale the input
    :return: scaled Numpy-Int-Array
    """
    return np.array(input) / maxIntValue


def decodeFromNumpyIntArray(input, maxIntValue):
    """
    Decode input value to unscaled Int
    :param input: Scaled Input-Value
    :param maxIntValue: Max-Value to unscale the input
    :return: unscaled Int
    """
    return input * maxIntValue


def createOutputDir(outputdir):
    """
    Create Folder in Filesystem
    :param outputdir: Folder-Name
    """
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)


###


class AlgorithmusTesterMT:
    def __init__(self, modelname, samples, batchsize, epochs, description=",",
                 network=CustomNetworkLayouts.defaultHiddenLayers, maxIntValue=sys.maxsize,
                 networkDataType=NetworkDataType.BIT, ciphertextAsInput=True):
        self.modelname = modelname
        self.samples = samples
        self.batchsize = batchsize
        self.epochs = epochs
        self.description = description
        self.network = network()
        self.maxIntValue = maxIntValue
        self.intBits = int(math.ceil(math.log(maxIntValue, 2)))
        self.networkDataType = networkDataType
        self.ciphertextAsInput = ciphertextAsInput
        self.outputdir = "training-files-{}-{}-S".format(self.modelname, self.samples)
        self.logdir = "logs/{}".format(self.modelname)
        self.time = time()

    def printSummaryToFile(self, lossfunction, history, stopped_epoch=-1):
        """
        Method to print a summaryfile of the training statics.
        :param lossfunction: lossfunction used
        :param history: history-object from the training
        :param stopped_epoch: epoch counter
        """
        data = ""
        for k, v in history.history.items():
            data = data + "{}:{};".format(k, v[-1])

        fields = [self.modelname, self.time, self.description, self.samples, self.batchsize, self.epochs,
                  lossfunction.__name__, stopped_epoch, self.network.getName(),
                  "{}_{}-{}".format(self.logdir, self.time, self.network.getName()), data]
        with open(r'{}.csv'.format(self.modelname), 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def encodeToNumpyArray(self, input):
        """
        Method to differentiate the encoding of the two data types.
        :param input: data
        :return: data encoded as Numpy-Array
        """
        if self.networkDataType == NetworkDataType.BIT:

            return encodeToNumpyBitArray(input, self.intBits)

        elif self.networkDataType == NetworkDataType.INT:

            return encodeToNumpyIntArray(input, self.maxIntValue)

    def decodeFromNumpyArray(self, input):
        """
        Method to differentiate the decoding of the two data types.
        :param input: data encoded as Numpy-Array
        :return: data
        """
        if self.networkDataType == NetworkDataType.BIT:
            target_length = math.ceil(len(input) / 8) * 8
            padded_input = np.zeros(target_length)
            padded_input[target_length - len(input):target_length] = input
            cipher_array_rd = np.around(padded_input)
            cipher_bytes = np.packbits(cipher_array_rd.astype(dtype=np.int))
            return cipher_bytes

        elif self.networkDataType == NetworkDataType.INT:

            return decodeFromNumpyIntArray(input, self.maxIntValue)

    def generate_training_data(self, randomFunc, encryptFunc, mkunique=True):
        """
        Methode to generate the training-data
        :param randomFunc: function to generate random-data
        :param encryptFunc: function to encrypt the random-data
        :param mkunique: Should the data used for the training be unique?
        :return: Save the generated data as encoded Numpy-Array in the filesystem
        """
        createOutputDir(self.outputdir)

        if self.ciphertextAsInput:
            outputshape = self.encodeToNumpyArray(randomFunc()).shape
            inputshape = self.encodeToNumpyArray(encryptFunc(randomFunc())).shape
        else:
            inputshape = self.encodeToNumpyArray(randomFunc()).shape
            outputshape = self.encodeToNumpyArray(encryptFunc(randomFunc())).shape

        input = np.zeros((self.samples, inputshape[0]), dtype=np.float)  # ciphertext matrix
        output = np.zeros((self.samples, outputshape[0]), dtype=np.float)  # plaintext matrix

        seen = set()

        print('Generating data...')
        i = 0
        while i < self.samples:
            plain = randomFunc()

            # if mkunique:
            #     key = bytes(plain) if self.networkDataType == NetworkDataType.BIT else ' '.join(str(x) for x in plain)
            #     if key in seen:
            #         continue
            #     seen.add(key)

            if mkunique:
                if plain in seen:
                    continue
                seen.add(plain)

            cipher = encryptFunc(plain)

            if self.ciphertextAsInput:
                output[i] = self.encodeToNumpyArray(plain)
                input[i] = self.encodeToNumpyArray(cipher)
            else:
                output[i] = self.encodeToNumpyArray(cipher)
                input[i] = self.encodeToNumpyArray(plain)

            if i % 1000 == 0:
                print("Generated: {} / {}".format(i, self.samples))

            i = i + 1

        print('Total questions:', len(input))

        print(input.shape)
        print(output.shape)
        print(input[1])
        print(output[1])

        indices = np.arange(len(output))
        np.random.shuffle(indices)
        input = input[indices]
        output = output[indices]

        split_at = len(input) - len(input) // 10
        (Input_train, Input_val) = input[:split_at], input[split_at:]
        (Output_train, Output_val) = output[:split_at], output[split_at:]

        print('Training Data:')
        print(Input_train.shape)
        print(Output_train.shape)

        print('Validation Data:')
        print(Input_val.shape)
        print(Output_val.shape)

        np.save("{}/Input_train.npy".format(self.outputdir), Input_train)
        np.save("{}/Input_val.npy".format(self.outputdir), Input_val)
        np.save("{}/Output_train.npy".format(self.outputdir), Output_train)
        np.save("{}/Output_val.npy".format(self.outputdir), Output_val)

    def train(self, lossfunction=losses.mean_squared_error, metrics=[CustomLosses.bitwise_loss, CustomLosses.full_loss],
              earlyStop=True, autotrain=False):
        """
        Method to train the neuronal Network with the generated trainingdata
        :param lossfunction: lossfunction used for the training
        :param metrics: custom metrics
        :param earlyStop: Should the training to early, if the results get worse?
        :param autotrain: Should the training run in autotrain-mode (separated folder and different-modelname)?
        :return: Save the trained model in the filesystem
        """

        Input_train = np.load("{}/Input_train.npy".format(self.outputdir))
        Input_val = np.load("{}/Input_val.npy".format(self.outputdir))
        Output_train = np.load("{}/Output_train.npy".format(self.outputdir))
        Output_val = np.load("{}/Output_val.npy".format(self.outputdir))

        print('Build model...')

        input_layer = layers.Input(shape=(Input_train.shape[1],))
        output_res = self.network.getNetwork(input_layer, Output_train.shape[1])

        model = Model(inputs=[input_layer], outputs=[output_res])

        model.compile(loss=lossfunction,
                      optimizer='adam',
                      metrics=metrics)
        model.summary()

        tf_callback = callbacks.TensorBoard(log_dir="{}_{}-{}".format(self.logdir, self.time, self.network.getName()))

        if self.networkDataType == NetworkDataType.BIT:
            es = callbacks.EarlyStopping(monitor='val_bitwise_loss',
                                         min_delta=0,
                                         patience=10,
                                         verbose=1, mode='auto')
        else:
            es = callbacks.EarlyStopping(monitor='val_loss',
                                         min_delta=0,
                                         patience=10,
                                         verbose=1, mode='auto')

        if earlyStop:
            history = model.fit(Input_train, Output_train,
                                batch_size=self.batchsize,
                                epochs=self.epochs,
                                validation_data=(Input_val, Output_val),
                                callbacks=[tf_callback, es])
            self.printSummaryToFile(lossfunction, history, es.stopped_epoch)

        else:
            history = model.fit(Input_train, Output_train,
                                batch_size=self.batchsize,
                                epochs=self.epochs,
                                validation_data=(Input_val, Output_val),
                                callbacks=[tf_callback])
            self.printSummaryToFile(lossfunction, history)

        if autotrain:
            createOutputDir("autotrain/")
            model.save("autotrain/{}_{}_{}.model".format(self.modelname, self.network.getName(), self.time))

        else:
            model.save("{}_{}.model".format(self.modelname, self.network.getName()))

    def evaluate(self, format_cipher=format_nothing, format_plain=format_nothing):
        """
        Method to evaluate the trained model
        :param format_cipher: function to format the ciphertext as string
        :param format_plain: function to format the plaintext as string
        :return: Print evaluation on console
        """
        Input_val = np.load("{}/Input_val.npy".format(self.outputdir))
        Output_val = np.load("{}/Output_val.npy".format(self.outputdir))

        model = load_model("{}_{}.model".format(self.modelname, self.network.getName()),
                           custom_objects={'bitwise_loss': CustomLosses.bitwise_loss,
                                           'full_loss': CustomLosses.full_loss,
                                           'doublesquare_error': CustomLosses.doublesquare_error})

        for i in range(80):
            ind = np.random.randint(0, len(Input_val))
            rowInput, rowOutput = Input_val[np.array([ind])], Output_val[np.array([ind])]
            preds = model.predict(rowInput, verbose=0)

            if self.ciphertextAsInput:
                q = format_cipher(self.decodeFromNumpyArray(rowInput[0]))
                correct = format_plain(self.decodeFromNumpyArray(rowOutput[0]))
                guess = format_plain(self.decodeFromNumpyArray(preds[0]))
            else:
                q = format_plain(self.decodeFromNumpyArray(rowInput[0]))
                correct = format_cipher(self.decodeFromNumpyArray(rowOutput[0]))
                guess = format_cipher(self.decodeFromNumpyArray(preds[0]))

            print('Question:', q, end=' ')
            print('Expected:', correct, end=' ')

            if isinstance(correct, (np.ndarray)):
                if np.array_equal(correct, guess):
                    print('Guess:', guess, end=' ')
                    print(bcolors.OKGREEN+'TRUE'+bcolors.ENDC, end='\n')

                else:
                    print('Guess:', guess, end=' ')
                    print(bcolors.FAIL+'FALSE'+bcolors.ENDC, end='\n')

            else:
                if correct == guess:
                    print('Guess:', guess, end=' ')
                    print(bcolors.OKGREEN+'TRUE'+bcolors.ENDC, end='\n')
                else:
                    print('Guess:', guess, end=' ')
                    print(bcolors.FAIL+'FALSE'+bcolors.ENDC, end='\n')

            #print(guess)
