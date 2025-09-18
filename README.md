# Scaling-Equilibrium-Propagation-to-Deeper-Neural-Network-Architectures
This repo contains the code experimets relating to the paper "Scaling Equilibrium Propagation to Deeper Neural Network Architectures" 

This code is based on the [rain-neuromorphics](https://github.com/rain-neuromorphics/energy-based-learning/tree/main) implementation of Centered Equilibrium Propagation, with changes made to support the experiments in this project.

# Instructions

Clone the repository:
```
git clone https://github.com/BrainSeek-Lab/Scaling-Equilibrium-Propagation-to-Deeper-Neural-Network-Architectures.git
cd Scaling-Equilibrium-Propagation-to-Deeper-Neural-Network-Architectures
```

Install dependencies:
```
pip install -r requirements.txt
```

The `EquiProp` directory contains the code for Equilibrium Propagation, while the `BackProp` directory includes the code used to create the feedforward equivalent of the network.

The code uses `wandb` for experiment tracking. 
