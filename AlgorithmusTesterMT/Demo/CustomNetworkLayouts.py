from keras import layers


#Network Definintion like the DES-Paper
class defaultHiddenLayers:
    def __init__(self):
        self.name = "CT_128sg-256sg-256sg-128sg"

    def getName(self):
        print(self.name)
        return self.name
    

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        hidden_1 = layers.Dense(128, activation='sigmoid')
        hidden_2 = layers.Dense(256, activation='sigmoid')
        hidden_3 = layers.Dense(256, activation='sigmoid')
        hidden_4 = layers.Dense(128, activation='sigmoid')

        hidden_1_res = hidden_1(input_layer)
        concat_1 = layers.concatenate([hidden_1_res, input_layer])
        hidden_2_res = hidden_2(concat_1)
        concat_2 = layers.concatenate([hidden_2_res, concat_1])
        hidden_3_res = hidden_3(concat_2)
        concat_3 = layers.concatenate([hidden_3_res, concat_2])
        hidden_4_res = hidden_4(concat_3)

        output_res = output_layer(hidden_4_res)

        return output_res


#des_simple_1559933757.1620996
class HiddenLayers_Simple:
    def __init__(self):
        self.name = "256sg-512sg-512sg-256sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        hidden_1 = layers.Dense(256, activation='sigmoid')
        hidden_2 = layers.Dense(512, activation='sigmoid')
        hidden_3 = layers.Dense(512, activation='sigmoid')
        hidden_4 = layers.Dense(256, activation='sigmoid')

        hidden_1_res = hidden_1(input_layer)
        hidden_2_res = hidden_2(hidden_1_res)
        hidden_3_res = hidden_3(hidden_2_res)
        hidden_4_res = hidden_4(hidden_3_res)

        output_res = output_layer(hidden_4_res)

        return output_res

#des_simple_1559934416.9970229
class HiddenLayers_five:
    def __init__(self):
        self.name = "CT_256sg-512sg-512sg-512sg-256sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        hidden_1 = layers.Dense(256, activation='sigmoid')
        hidden_2 = layers.Dense(512, activation='sigmoid')
        hidden_3 = layers.Dense(512, activation='sigmoid')
        hidden_4 = layers.Dense(512, activation='sigmoid')
        hidden_5 = layers.Dense(256, activation='sigmoid')

        hidden_1_res = hidden_1(input_layer)
        concat_1 = layers.concatenate([hidden_1_res, input_layer])
        hidden_2_res = hidden_2(concat_1)
        concat_2 = layers.concatenate([hidden_2_res, concat_1])
        hidden_3_res = hidden_3(concat_2)
        concat_3 = layers.concatenate([hidden_3_res, concat_2])
        hidden_4_res = hidden_4(concat_3)
        concat_4 = layers.concatenate([hidden_4_res, concat_3])
        hidden_5_res = hidden_5(concat_4)

        output_res = output_layer(hidden_5_res)

        return output_res


class defaultHiddenLayers_initbias:
    def __init__(self):
        self.name = "CT-IB_128sg-256sg-256sg-128sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        hidden_1 = layers.Dense(128, activation='sigmoid', bias_initializer='glorot_uniform')
        hidden_2 = layers.Dense(256, activation='sigmoid', bias_initializer='glorot_uniform')
        hidden_3 = layers.Dense(256, activation='sigmoid', bias_initializer='glorot_uniform')
        hidden_4 = layers.Dense(128, activation='sigmoid', bias_initializer='glorot_uniform')

        hidden_1_res = hidden_1(input_layer)
        concat_1 = layers.concatenate([hidden_1_res, input_layer])
        hidden_2_res = hidden_2(concat_1)
        concat_2 = layers.concatenate([hidden_2_res, concat_1])
        hidden_3_res = hidden_3(concat_2)
        concat_3 = layers.concatenate([hidden_3_res, concat_2])
        hidden_4_res = hidden_4(concat_3)

        output_res = output_layer(hidden_4_res)

        return output_res

class defaultHiddenLayers_initbias_re:
    def __init__(self):
        self.name = "CT-IB_128re-256re-256re-128re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')

        hidden_1 = layers.Dense(128, activation='relu', bias_initializer='glorot_uniform')
        hidden_2 = layers.Dense(256, activation='relu', bias_initializer='glorot_uniform')
        hidden_3 = layers.Dense(256, activation='relu', bias_initializer='glorot_uniform')
        hidden_4 = layers.Dense(128, activation='relu', bias_initializer='glorot_uniform')

        hidden_1_res = hidden_1(input_layer)
        concat_1 = layers.concatenate([hidden_1_res, input_layer])
        hidden_2_res = hidden_2(concat_1)
        concat_2 = layers.concatenate([hidden_2_res, concat_1])
        hidden_3_res = hidden_3(concat_2)
        concat_3 = layers.concatenate([hidden_3_res, concat_2])
        hidden_4_res = hidden_4(concat_3)

        output_res = output_layer(hidden_4_res)

        return output_res


class genHiddenLayers_initbias_five:
    def __init__(self):
        self.name = "CT-IB_5x1024sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        pref_layer = input_layer
        for i in range(4):
            hidden = layers.Dense(1024, activation='sigmoid', bias_initializer='glorot_uniform')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_initbias_eight:
    def __init__(self):
        self.name = "CT-IB_8x1024sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        pref_layer = input_layer
        for i in range(7):
            hidden = layers.Dense(1024, activation='sigmoid', bias_initializer='glorot_uniform')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_initbias_six:
    def __init__(self):
        self.name = "CT-IB_5x1024sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        pref_layer = input_layer
        for i in range(5):
            hidden = layers.Dense(1024, activation='sigmoid', bias_initializer='glorot_uniform')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res


class genHiddenLayers_hundert:
    def __init__(self):
        self.name = "CT-IB_100x1024sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        pref_layer = input_layer
        for i in range(100):
            hidden = layers.Dense(128, activation='sigmoid')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res


class genLayers_zwanzig:
    def __init__(self):
        self.name = "IB_20x32sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='sigmoid')

        pref_layer = input_layer
        for i in range(20):
            hidden = layers.Dense(32, activation='sigmoid')
            pref_layer = hidden(pref_layer)

        output_res = output_layer(pref_layer)

        return output_res

class genLayers_TwoLayers:
    def __init__(self):
        self.name = "IB_2x32re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')
        pref_layer = input_layer
        for i in range(2):
            hidden = layers.Dense(32, activation='relu')
            pref_layer = hidden(pref_layer)

        output_res = output_layer(pref_layer)

        return output_res

class genLayers_OneLayers:
    def __init__(self):
        self.name = "IB_1x10re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')
        pref_layer = input_layer
        for i in range(1):
            hidden = layers.Dense(10, activation='relu')
            pref_layer = hidden(pref_layer)

        output_res = output_layer(pref_layer)

        return output_res

class genLayers_FiveLayers:
    def __init__(self):
        self.name = "IB_5x32sg"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')
        pref_layer = input_layer
        for i in range(16):
            hidden = layers.Dense(128, activation='relu')
            pref_layer = hidden(pref_layer)

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_six:
    def __init__(self):
        self.name = "CT_6x256re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')

        pref_layer = input_layer
        for i in range(6):
            hidden = layers.Dense(256, activation='relu')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_ten:
    def __init__(self):
        self.name = "CT_10x256re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')

        pref_layer = input_layer
        for i in range(10):
            hidden = layers.Dense(256, activation='relu')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_twenty:
    def __init__(self):
        self.name = "CT_20x128re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')

        pref_layer = input_layer
        for i in range(20):
            hidden = layers.Dense(128, activation='relu')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

class genHiddenLayers_five:
    def __init__(self):
        self.name = "CT_5x128re"

    def getName(self):
        return self.name

    def getNetwork(self, input_layer, outputShape):

        output_layer = layers.Dense(outputShape, activation='relu')

        pref_layer = input_layer
        for i in range(5):
            hidden = layers.Dense(128, activation='relu')
            hidden_res = hidden(pref_layer)
            pref_layer = layers.concatenate([hidden_res, pref_layer])

        output_res = output_layer(pref_layer)

        return output_res

#Network Definintion like the DES-Paper
# def defaultHiddenLayers(input_layer):
#
#     hidden_1 = layers.Dense(128, activation='sigmoid')
#     hidden_2 = layers.Dense(256, activation='sigmoid')
#     hidden_3 = layers.Dense(256, activation='sigmoid')
#     hidden_4 = layers.Dense(128, activation='sigmoid')
#
#     hidden_1_res = hidden_1(input_layer)
#     concat_1 = layers.concatenate([hidden_1_res, input_layer])
#     hidden_2_res = hidden_2(concat_1)
#     concat_2 = layers.concatenate([hidden_2_res, concat_1])
#     hidden_3_res = hidden_3(concat_2)
#     concat_3 = layers.concatenate([hidden_3_res, concat_2])
#     hidden_4_res = hidden_4(concat_3)
#
#     return (name, hidden_4_res)