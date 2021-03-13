"""
from: https://github.com/keras-team/keras-io/blob/master/examples/structured_data/structured_data_classification_from_scratch.py
Title: Structured data classification from scratch
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/06/09
Last modified: 2020/06/09
Description: Binary classification of structured data including numerical and categorical features.
"""
"""
## Introduction
This example demonstrates how to do structured data classification, starting from a raw
CSV file. Our data includes both numerical and categorical features. We will use Keras
preprocessing layers to normalize the numerical features and vectorize the categorical
ones.
Note that this example should be run with TensorFlow 2.3 or higher, or `tf-nightly`.
### The dataset
[Our dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease) is provided by the
Cleveland Clinic Foundation for Heart Disease.
It's a CSV file with 303 rows. Each row contains information about a patient (a
**sample**), and each column describes an attribute of the patient (a **feature**). We
use the features to predict whether a patient has a heart disease (**binary
classification**).
Here's the description of each feature:
Column| Description| Feature Type
------------|--------------------|----------------------
Age | Age in years | Numerical
Sex | (1 = male; 0 = female) | Categorical
CP | Chest pain type (0, 1, 2, 3, 4) | Categorical
Trestbpd | Resting blood pressure (in mm Hg on admission) | Numerical
Chol | Serum cholesterol in mg/dl | Numerical
FBS | fasting blood sugar in 120 mg/dl (1 = true; 0 = false) | Categorical
RestECG | Resting electrocardiogram results (0, 1, 2) | Categorical
Thalach | Maximum heart rate achieved | Numerical
Exang | Exercise induced angina (1 = yes; 0 = no) | Categorical
Oldpeak | ST depression induced by exercise relative to rest | Numerical
Slope | Slope of the peak exercise ST segment | Numerical
CA | Number of major vessels (0-3) colored by fluoroscopy | Both numerical & categorical
Thal | 3 = normal; 6 = fixed defect; 7 = reversible defect | Categorical
Target | Diagnosis of heart disease (1 = true; 0 = false) | Target
"""

"""
## Setup
"""

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers


"""
## Preparing the data
Let's download the data and load it into a Pandas dataframe:
"""

file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
dataframe = pd.read_csv(file_url)

"""
The dataset includes 303 samples with 14 columns per sample (13 features, plus the target
label):
"""

dataframe.shape

"""
Here's a preview of a few samples:
"""

dataframe.head()

"""
The last column, "target", indicates whether the patient has a heart disease (1) or not
(0).
Let's split the data into a training and validation set:
"""

val_dataframe = dataframe.sample(frac=0.2, random_state=1337)
train_dataframe = dataframe.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

"""
Let's generate `tf.data.Dataset` objects for each dataframe:
"""


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

"""
Each `Dataset` yields a tuple `(input, target)` where `input` is a dictionary of features
and `target` is the value `0` or `1`:
"""

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

"""
Let's batch the datasets:
"""

train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)

"""
## Feature preprocessing with Keras layers
The following features are categorical features encoded as integers:
- `sex`
- `cp`
- `fbs`
- `restecg`
- `exang`
- `ca`
We will encode these features using **one-hot encoding** using the `CategoryEncoding()`
layer.
We also have a categorical feature encoded as a string: `thal`. We will first create an
index of all possible features using the `StringLookup()` layer, then we will one-hot
encode the output indices using a `CategoryEncoding()` layer.
Finally, the following feature are continuous numerical features:
- `age`
- `trestbps`
- `chol`
- `thalach`
- `oldpeak`
- `slope`
For each of these features, we will use a `Normalization()` layer to make sure the mean
of each feature is 0 and its standard deviation is 1.
Below, we define 3 utility functions to do the operations:
- `encode_numerical_feature` to apply featurewise normalization to numerical features.
- `encode_string_categorical_feature` to first turn string inputs into integer indices,
then one-hot encode these integer indices.
- `encode_integer_categorical_feature` to one-hot encode integer categorical features.
"""

from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature


"""
## Build a model
With this done, we can create our end-to-end model:
"""

# Categorical features encoded as integers
sex = keras.Input(shape=(1,), name="sex", dtype="int64")
cp = keras.Input(shape=(1,), name="cp", dtype="int64")
fbs = keras.Input(shape=(1,), name="fbs", dtype="int64")
restecg = keras.Input(shape=(1,), name="restecg", dtype="int64")
exang = keras.Input(shape=(1,), name="exang", dtype="int64")
ca = keras.Input(shape=(1,), name="ca", dtype="int64")

# Categorical feature encoded as string
thal = keras.Input(shape=(1,), name="thal", dtype="string")

# Numerical features
age = keras.Input(shape=(1,), name="age")
trestbps = keras.Input(shape=(1,), name="trestbps")
chol = keras.Input(shape=(1,), name="chol")
thalach = keras.Input(shape=(1,), name="thalach")
oldpeak = keras.Input(shape=(1,), name="oldpeak")
slope = keras.Input(shape=(1,), name="slope")

all_inputs = [
    sex,
    cp,
    fbs,
    restecg,
    exang,
    ca,
    thal,
    age,
    trestbps,
    chol,
    thalach,
    oldpeak,
    slope,
]

# Integer categorical features
sex_encoded = encode_integer_categorical_feature(sex, "sex", train_ds)
cp_encoded = encode_integer_categorical_feature(cp, "cp", train_ds)
fbs_encoded = encode_integer_categorical_feature(fbs, "fbs", train_ds)
restecg_encoded = encode_integer_categorical_feature(restecg, "restecg", train_ds)
exang_encoded = encode_integer_categorical_feature(exang, "exang", train_ds)
ca_encoded = encode_integer_categorical_feature(ca, "ca", train_ds)

# String categorical features
thal_encoded = encode_string_categorical_feature(thal, "thal", train_ds)

# Numerical features
age_encoded = encode_numerical_feature(age, "age", train_ds)
trestbps_encoded = encode_numerical_feature(trestbps, "trestbps", train_ds)
chol_encoded = encode_numerical_feature(chol, "chol", train_ds)
thalach_encoded = encode_numerical_feature(thalach, "thalach", train_ds)
oldpeak_encoded = encode_numerical_feature(oldpeak, "oldpeak", train_ds)
slope_encoded = encode_numerical_feature(slope, "slope", train_ds)

all_features = layers.concatenate(
    [
        sex_encoded,
        cp_encoded,
        fbs_encoded,
        restecg_encoded,
        exang_encoded,
        slope_encoded,
        ca_encoded,
        thal_encoded,
        age_encoded,
        trestbps_encoded,
        chol_encoded,
        thalach_encoded,
        oldpeak_encoded,
    ]
)


# Pennylane / Qiskit code for quantum layer: 1 node, input-vector-length = 4 (TODO: make 16)
import pennylane as qml
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import MCPhaseGate
import matplotlib

# 2 regular qubits, 1 ancilla
number_qubits = 3
number_regular_qubits = number_qubits - 1
size = number_regular_qubits ** 2

# specify device with Pennylane
dev = qml.device("default.qubit", wires=number_qubits)


# specify qiskit circuit
# specify parameters to set later with NN inputs and weights using Keras + Pennylane
#inputs = []
#weights = []
#for s in range(size):
    #inputs.append(Parameter('θ' + str(s)))
    #weights.append(Parameter('ϕ' + str(s)))
theta1 = Parameter('θ1')
theta2 = Parameter('θ2')
theta3 = Parameter('θ3')
w1 = Parameter('ϕ1')
w2 = Parameter('ϕ2')
w3 = Parameter('ϕ3')
# TODO: Normalize weights and inputs to be between [0, pi / 2]
# TODO: Figure out how MCPhaseGate works -> replace CPhase-Gate (cu1) with this and Toffoli with Multi-controlled CNOT and use 4 qubits, Input-Size 16 instead
qc = QuantumCircuit(number_qubits)
# apply Hadamard gate to all regular qubits to create a superposition
for i in range(number_regular_qubits):
    qc.h(i)
qc.barrier()
# loop over all inputs in inputvector to encode them to the right base states using phase-shifts
for index in range(1, size):
    if index == 1:
        value = theta1
    elif index == 2:
        value = theta2
    elif index == 3:
        value = theta3
    # index as binary number
    binary = '{0:02b}'.format(index)
    # get qubit at digit in binary state (positions of qubits : q0, q1, q2, q3) (figuratively, not actually, we are in superposition after all)
    for j in range(len(binary)):
        if binary[j] == '0':
            qc.x(j)
    # this_phase_gate = MCPhaseGate(value, 3, label="this_phase_gate")
    # qc.this_phase_gate(0, 1, 2, 3)
    # perform controlled phase shift
    qc.cp(value, 0, 1)
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            qc.x(j)
    qc.barrier()
# loop over weights
for w in range(1, size):
    if w == 1:
        value = w1
    elif w == 2:
        value = w2
    elif w == 3:
        value = w3
    # index as binary number
    binary = '{0:02b}'.format(w)
    # get qubit at digit in binary state (positions of qubits : q0, q1, q2, q3) (figuratively, not actually, we are in superposition after all)
    for j in range(len(binary)):
        if binary[j] == '0':
            qc.x(j)
    # this_phase_gate = MCPhaseGate(value, 3, label="this_phase_gate")
    # qc.this_phase_gate(0, 1, 2, 3)
    # perform controlled phase shift
    qc.cp(value, 0, 1).inverse()
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            qc.x(j)
    qc.barrier()
# apply Hadamard gate to all regular qubits
for i in range(number_regular_qubits):
    qc.h(i)
qc.barrier()
# apply X gate to all regular qubits
for i in range(number_regular_qubits):
    qc.x(i)
qc.barrier()
# collect combined state from all regular qubits with ancilla qubit using multi-controlled NOT-gate (Toffoli-Gate in case of 2 regular qubits)
qc.ccx(0, 1, 2)
# draw circuit
qc.draw(output='mpl', filename="circuit.png", plot_barriers=False)
# end qiskit circuit


weight_shapes = {"weights": (size - 1)}
# index of ancilla-qubit
ancilla = number_qubits - 1


@qml.qnode(dev)
def qnode(inputs, weights):
    inputs = inputs.numpy()
    weights = weights.numpy()
    input_list = []
    for i in range(1, size):
        input_list.append(inputs[i] - inputs[0])
    # run qiskit circuit
    qc.bind_parameters({theta1: input_list[0], theta2: input_list[1], theta3: input_list[2], w1: weights[0], w2: weights[1], w3: weights[2]})
    qml.from_qiskit(qc)
    # measure ancilla-qubit
    # TODO: Try out whether expectation value (expval) or sample value (sample) of the qubit works better
    return qml.expval(qml.PauliZ(ancilla))


# create qlayer
qlayer = qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=1)
qlayer.build(size)


x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
x = layers.Dense(4, activation="relu")(x)
x = layers.Dropout(0.5)(x)
output = qlayer(x)
#output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
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

model.fit(train_ds, epochs=50, validation_data=val_ds)

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

sample = {
    "age": 60,
    "sex": 1,
    "cp": 1,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 2,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 3,
    "ca": 0,
    "thal": "fixed",
}

input_dict = {name: tf.convert_to_tensor([value]) for name, value in sample.items()}
predictions = model.predict(input_dict)

print(
    "This particular patient had a %.1f percent probability "
    "of having a heart disease, as evaluated by our model." % (100 * predictions[0][0],)
)