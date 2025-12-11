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

Run the code as 
```
python EquEquiProp/dhcnresnet.py --dataset CIFAR10 --epochs 300
```



### Command-line Arguments

| Argument                     | Type  | Default              | Description                                                    | Choices / Notes                                                                                  |
| ---------------------------- | ----- | -------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `--dataset`                  | str   | `'FashionMNIST'`     | Dataset used for training                                      | `'CIFAR10'`, `'CIFAR100'`, `'FashionMNIST'`                                                      |
| `--epochs`                   | int   | `100`                | Number of training epochs                                      | —                                                                                                |
| `--activation`               | str   | `'relu6'`            | Activation function used in the network                        | `'hard-sigmoid'`, `'relu6'`, `'reluclip'`                                              |
| `--lossfn`                   | str   | `'MSE'`              | Loss function                                                  | `'MSE'` (mean squared error), `'CE'` (cross-entropy)                                             |
| `--optimizer`                | str   | `'sgd'`              | Optimizer for training                                         | `'sgd'` (NAG version), `'adam'` (for Adam, use learning rate 1e-4) |
| `--use_warm_restarts`        | flag  | `False`              | Use cosine annealing with warm restarts for learning rates     | If not set, uses standard cosine annealing                                                       |
| `--beta`                     | float | `0.1`                | Nudging factor                                                 | Best found: CIFAR10=0.1, CIFAR100=0.15, FashionMNIST=0.15                                        |
| `--num_iterations_inference` | int   | `120`                | Number of inference iterations                                 | —                                                                                                |
| `--num_iterations_training`  | int   | `50`                 | Number of training iterations                                  | —                                                                                                |
| `--weight_gains`             | str   | `None`               | Weight gains as comma-separated floats (one per weight matrix) | 13 values for HopfieldResNet13, 5 for VGG5                                                       |
| `--learning_rates_weights`   | str   | `None`               | Learning rates for weights as comma-separated floats           | One per weight matrix                                                                            |
| `--learning_rates_biases`    | str   | `None`               | Learning rates for biases as comma-separated floats            | One per weight matrix                                                                            |
| `--momentum`                 | float | `0.9`                | Momentum factor for SGD optimizer                              | Best found 0.9                                                                                   |
| `--weight_decay`             | float | `3.5e-4`             | Weight decay factor for SGD optimizer                          | Best found: CIFAR10=2.5e-4, CIFAR100=3.5e-4, FashionMNIST=3.5e-4                                 |
| `--batch_size`               | int   | `128`                | Batch size for training                                        | —                                                                                                |
| `--network`                  | str   | `'hopfieldresnet13'` | Network architecture                                           | `'hopfieldresnet13'`, `'vgg5'`                                                                   |
| `--no_wandb`                 | flag  | `False`              | Disable Weights & Biases logging                               | Only print results in console                                                                    |

---

If anyone want to use the code pleae cite our work,

```
@misc{scalingequilibriumpropagationdeeper,
      title={Scaling Equilibrium Propagation to Deeper Neural Network Architectures}, 
      author={Sankar Vinayak Elayedam and Gopalakrishnan Srinivasan},
      year={2025},
      eprint={2509.26003},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/2509.26003}, 
}
```
