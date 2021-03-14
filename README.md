# QDynamics
This is the repository of our project "QDynamics" (part of the PushQuantum "Quantum Entrepreneurship Laboratory").
In this Project, we aim to modify a Physics Inspired Neural Network (PINN) to use a quantum layer. This proof of concept intends to show that it is possible to use Quantum Neural Network layers in Neural Networks that solve Partial Differential Equations (PDEs). With these Hybrid Quantum-Classical Neural Networks, it will most likely be possible to speed up Computational Fluid Dynamics (CFD) calculations in the future.

## Front-End
A Clickable Prototype of a Front-End for our CFD application using Figma can be found [here](https://www.figma.com/proto/tJQ0RjgjVYJ0hsxPaGyfCf/QDynamics-UI-Prototype?node-id=6%3A567&scaling=min-zoom).

## PINN and QPINN
A classical implementation of a Physics Inspired Neural Network (PINN) solving the Burger's Equation (which we adapted from https://github.com/pierremtb/PINNs-TF2.0 ), as well as our modification of this network to incorporate a quantum layer, can be found in the following [Google Colab notebook](https://colab.research.google.com/drive/1swzm2eV_Ig8iJK2G1NLB-eatCGk-DW-6?usp=sharing#scrollTo=UTnQGzs2_6-3). 

The quantum layer found in the hybrid quantum neural network is structured as seen in the figure below:
![Quantum Layer Circuit](circuit.png?raw=true "Quantum Layer Circuit Representation")

We implemented the QPINN using Tensorflow Quantum. Unfortunately, since this framwork – like all Quantum Machine Learning Frameworks – is still very young, it does not provide the possibility to calculate higher-order gradients on quantum layers yet, which our application would require. Other people have also been experiencing this [issue](https://github.com/tensorflow/quantum/issues/431 ), so a corresponding feature will hopefully be provided soon. For this reason, our Quantum Physics-Inspired Neural Network (QPINN) is not entirely executable just yet.

## Quantum Layer
To show that our Quantum Layer is most defininetely executable per se, we additionally provided a toy example using it in another neural network:
keras_example_tfq.ipynb (or, as a Google Collab: https://colab.research.google.com/drive/1ngD5YMM7EeWnWxXgMZfk8GgXPdhSoexI)

## Results
For the classical version of the algorithm, the result is as follows:

![Result](plot.png?raw=true "Experimental Result")

