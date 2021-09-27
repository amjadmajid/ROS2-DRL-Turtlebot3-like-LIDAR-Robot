from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import numpy as np
from time import time

def timeit(func, arg, iterations):
    t0 = time()
    for _ in range(iterations):
        func(arg)
    print("%.4f sec" % (time() - t0))

ipt   = Input(shape=(4,))
x     = Dense(2, activation='relu')(ipt)
out   = Dense(1, activation='sigmoid')(x)
model = Model(ipt, out)

X = np.random.randn(32,4)

timeit(model.predict, X, 1000)
model.compile('adam', loss='binary_crossentropy')
timeit(model.predict, X, 1000)
model._make_train_function()  # build optimizer
timeit(model.predict, X, 1000)