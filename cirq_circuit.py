# TFQ / Cirq code for quantum layer: 1 node, input-vector-length = 4 (TODO: make 16)
#import tensorflow_quantum as tfq
import cirq
from cirq import H, X, cphase, CCX, measure
from cirq.circuits import InsertStrategy
import sympy
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit

# 2 regular qubits, 1 ancilla
number_qubits = 3
number_regular_qubits = number_qubits - 1
size = number_regular_qubits ** 2

# specify device with cirq
simulator = cirq.Simulator()

# specify cirq circuit
# specify parameters to set later with NN inputs and weights using Keras + TFQ
control_params = sympy.symbols('theta1 theta2 theta3 w1 w2 w3')
# TODO: Normalize weights and inputs to be between [0, pi / 2]
# TODO: Figure out how MCPhaseGate works -> replace CPhase-Gate (cu1) with this and Toffoli with Multi-controlled CNOT and use 4 qubits, Input-Size 16 instead
regular_qubits = cirq.LineQubit.range(number_regular_qubits)
ancilla = cirq.NamedQubit('ancilla')
qc = cirq.Circuit()
# apply Hadamard gate to all regular qubits to create a superposition
qc.append(H.on_each(*regular_qubits))
# loop over all inputs in inputvector to encode them to the right base states using phase-shifts
for index in range(1, size):
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
    insert_list.append(cphase(control_params[index - 1])(*regular_qubits))
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    qc.append(insert_list, strategy=InsertStrategy.NEW_THEN_INLINE)
# loop over weights
for w in range(1, size):
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
    insert_list.append(cphase(-1 * control_params[w + 2])(*regular_qubits))
    # "undo" the NOT-gates to get back to previous states = apply another not
    for j in range(len(binary)):
        if binary[j] == '0':
            insert_list.append(X(regular_qubits[j]))
    qc.append(insert_list, strategy=InsertStrategy.NEW_THEN_INLINE)
# apply Hadamard gate to all regular qubits
qc.append(H.on_each(*regular_qubits), strategy=InsertStrategy.NEW_THEN_INLINE)
# apply X gate to all regular qubits
qc.append(X.on_each(*regular_qubits), strategy=InsertStrategy.NEW_THEN_INLINE)
# collect combined state from all regular qubits with ancilla qubit using multi-controlled NOT-gate (Toffoli-Gate in case of 2 regular qubits)
qc.append(CCX(regular_qubits[0], regular_qubits[1], ancilla), strategy=InsertStrategy.NEW_THEN_INLINE)
# draw circuit
SVGCircuit(qc)