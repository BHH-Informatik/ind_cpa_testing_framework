import keras.backend as K

def bitwise_loss(y_true, y_pred): #Wie viele Prozent der Bit sind richtig?
    y_pred_rounded = K.round(y_pred)
    return K.not_equal(y_true, y_pred_rounded)

# def bytewise_loss(y_true, y_pred): #Wie viele Prozent der Bytes sind richtig?
#     y_pred_rounded = K.round(y_pred)
#     return K.not_equal(y_true, y_pred_rounded)

#custom_error
def full_loss(y_true, y_pred): #Wie viele Prozent sind komplett richtig?
    y_pred_rounded = K.round(y_pred)
    not_equal = K.not_equal(y_true, y_pred_rounded)
    return K.any(not_equal, axis=1)

def bitwise_loss_max_min(y_true, y_pred): #Wie viele Prozent der Bit sind richtig? => Ableitbar
    # round numbers less than 0.5 to zero;
    # by making them negative and taking the maximum with 0
    differentiable_round = K.maximum(y_pred-0.499, 0)
    # scale the remaining numbers (0 to 0.5) to greater than 1
    # the other half (zeros) is not affected by multiplication
    differentiable_round = differentiable_round * 10000
    # take the minimum with 1
    differentiable_round = K.minimum(differentiable_round, 1)
    square = K.square(y_true - differentiable_round)
    return K.mean(square)

def doublesquare_error(y_true, y_pred): #Wie viele Prozent der Bit sind richtig? => Ableitbar
    doublesquare = K.square(K.square(y_true - y_pred))
    return K.mean(doublesquare)