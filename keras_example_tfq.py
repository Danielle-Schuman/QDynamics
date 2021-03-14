"""
from: https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_from_scratch.py
Title: Structured data classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/06/09
Last modified: 2020/06/09
Description: Binary classification of structured data including numerical and categorical features.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# TFQ / Cirq code for quantum layer: 1 node, input-vector-length = 4 (TODO: make 16)
import tensorflow_quantum as tfq
import cirq
from cirq import H, X, cphase, CNOT, Z, T
from cirq.circuits import InsertStrategy
import sympy
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# adapted from https://github.com/ghellstern/QuantumNN/blob/master/Multi-QBit-Classifier%20TF%20NN-Encoding_Github.ipynb
class SplitBackpropQ(tf.keras.layers.Layer):

    def __init__(self, upstream_symbols, managed_symbols, managed_init_vals,
                 operators):
        """Create a layer that splits backprop between several variables.


        Args:
            upstream_symbols: Python iterable of symbols to bakcprop
                through this layer.
            managed_symbols: Python iterable of symbols to backprop
                into variables managed by this layer.
            managed_init_vals: Python iterable of initial values
                for managed_symbols.
            operators: Python iterable of operators to use for expectation.

        """
        super().__init__(SplitBackpropQ)
        self.all_symbols = upstream_symbols + managed_symbols
        self.upstream_symbols = upstream_symbols
        self.managed_symbols = managed_symbols
        self.managed_init = managed_init_vals
        self.ops = operators

    def build(self, input_shape):
        self.managed_weights = self.add_weight(
            shape=(1, len(self.managed_symbols)),
            initializer=tf.constant_initializer(self.managed_init))

    def call(self, inputs):
        # inputs are: circuit tensor, upstream values
        upstream_shape = tf.gather(tf.shape(inputs[0]), 0)
        tiled_up_weights = tf.tile(self.managed_weights, [upstream_shape, 1])
        joined_params = tf.concat([inputs[1], tiled_up_weights], 1)
        return tfq.layers.Expectation()(inputs[0],
                                        operators=measurement,
                                        symbol_names=self.all_symbols,
                                        symbol_values=joined_params)


# TODO: Normalize weights and inputs to be between [0, pi / 2]
# TODO: replace CPhase-Gate with Multi-controlled Phase gate and use 4 qubits, Input-Size 16 instead
# 2 regular qubits, 1 ancilla
number_qubits = 3
number_regular_qubits = number_qubits - 1

# specify parameters to set later with NN inputs and weights using Keras + TFQ
regular_qubits = [cirq.GridQubit(i, 0) for i in range(number_regular_qubits)]
ancilla = cirq.GridQubit(number_regular_qubits, 0)
control_params = sympy.symbols('i0, i1, i2, i3')
control_params1 = sympy.symbols('w0, w1, w2, w3')

# specify cirq circuit
qc = cirq.Circuit()
size = len(regular_qubits) ** 2
# subtract first input from other inputs to save gates
#inputs = []
#for i in range(1, size):
    #inputs.append(control_params[i] - control_params[0])
# do the same for weights
#weights = []
#for i in range(1, size):
    #weights.append(control_params1[i] - control_params1[0])
# apply Hadamard gate to all regular qubits to create a superposition
qc.append(H.on_each(*regular_qubits))
# loop over all inputs in inputvector to encode them to the right base states using phase-shifts
for index in range(size):
    insert_list = []
    # index as binary number
    binary = '{0:02b}'.format(index)
    # get qubit at digit in binary state (positions of qubits : q0, q1, q2, q3) (figuratively, not actually, we are in superposition after all)
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    # this_phase_gate = MCPhaseGate(value, 3, label="this_phase_gate")
    # qc.this_phase_gate(0, 1, 2, 3)
    # perform controlled phase shift (for more qubits probably possible using ControlledGate() and MatrixGate()
    insert_list.append(cphase(control_params[index])(*regular_qubits))
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    qc.append(insert_list, strategy=InsertStrategy.NEW_THEN_INLINE)
# loop over weights
for w in range(size):
    insert_list = []
    # index as binary number
    binary = '{0:02b}'.format(w)
    # get qubit at digit in binary state (positions of qubits : q0, q1, q2, q3) (figuratively, not actually, we are in superposition after all)
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    # this_phase_gate = MCPhaseGate(value, 3, label="this_phase_gate")
    # qc.this_phase_gate(0, 1, 2, 3)
    # perform conjugate transpose controlled phase shift
    insert_list.append(cphase((-1) * control_params1[w])(*regular_qubits))
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    qc.append(insert_list, strategy=InsertStrategy.NEW_THEN_INLINE)
# apply Hadamard gate to all regular qubits
qc.append(H.on_each(*regular_qubits), strategy=InsertStrategy.NEW_THEN_INLINE)
# apply X gate to all regular qubits
qc.append(X.on_each(*regular_qubits), strategy=InsertStrategy.NEW_THEN_INLINE)
# collect combined state from all regular qubits with ancilla qubit using Toffoli-Gate
# Toffoli-Gate does not work in TFQ -> implement decomposition (compare https://en.wikipedia.org/wiki/Toffoli_gate#/media/File:Qcircuit_ToffolifromCNOT.svg)
qc.append([H(ancilla), CNOT(regular_qubits[1], ancilla), cirq.inverse(T(ancilla)), CNOT(regular_qubits[0], ancilla), T(ancilla), CNOT(regular_qubits[1], ancilla), cirq.inverse(T(ancilla)), CNOT(regular_qubits[0], ancilla), T(ancilla), T(regular_qubits[1]), H(ancilla), CNOT(regular_qubits[0], regular_qubits[1]), cirq.inverse(T(regular_qubits[1])), T(regular_qubits[0]), CNOT(regular_qubits[0], regular_qubits[1])], strategy=InsertStrategy.NEW_THEN_INLINE)
# draw circuit
SVGCircuit(qc)
# end circuit

# values to initialize the weights (?)
np.random.seed(seed=69)
int_values = np.random.rand((len(control_params1)))*np.pi

measurement = [Z(ancilla)]

# This is needed because of Note here:
# https://www.tensorflow.org/quantum/api_docs/python/tfq/layers/Expectation
unused = tf.keras.Input(shape=(), dtype=tf.dtypes.string)
inputlayer = keras.Input(shape=(32,), name="inputlayer")
x = layers.Dense(32, activation="relu")(inputlayer)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4, activation="relu")(x)
x = layers.Dropout(0.5)(x)
expectation = SplitBackpropQ(control_params, control_params1, int_values, measurement)([unused, x])
#expectation = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=[unused, inputlayer], outputs=expectation)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

"""
Let's visualize our connectivity graph:
"""

# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
model.summary()
"""
## Train the model
"""
x_train = []
y_train = []
for _ in range(100):
    x_new = []
    for i in range(32):
        v = np.random.random_sample()
        x_new.append(v)
    x_train.append(x_new)
    y_train.append(np.rint(np.array(x_new).mean()))
x_train = tf.convert_to_tensor(x_train, tf.double)
y_train = tf.convert_to_tensor(y_train, tf.double)

x_val = []
y_val = []
for _ in range(100):
    x_new = []
    for i in range(32):
        v = np.random.random_sample()
        x_new.append(v)
    x_val.append(x_new)
    y_val.append(np.rint(np.array(x_new).mean()))
x_val = tf.convert_to_tensor(x_val, tf.double)
y_val = tf.convert_to_tensor(y_val, tf.double)

#n = train_ds.cardinality().numpy()
#n_val = val_ds.cardinality().numpy()

n = x_train.shape[0]
n_val = x_val.shape[0]

model.fit(x=[tfq.convert_to_tensor([qc for _ in range(n)]), x_train], y=y_train, epochs=100, batch_size=32, validation_data=([tfq.convert_to_tensor([qc for _ in range(n_val)]), x_val], y_val), validation_batch_size=32)

"""
We quickly get to 80% validation accuracy.
"""

"""
## Inference on new data
To get a prediction for a new sample, you can simply call `model.predict()`. There are
just two things you need to do:
1. wrap scalars into a list so as to have a batch dimension (models only process batches
of data, not single samples)
2. Call `convert_to_tensor` on each feature
"""

sample = tf.convert_to_tensor([[np.random.random_sample() for _ in range(32)]], tf.double)
predictions = model.predict([tfq.convert_to_tensor([qc]), sample])

print("Sample " + str(sample) + " label " + str(predictions) + " should be " + str(np.rint(np.array(sample[0]).mean())))