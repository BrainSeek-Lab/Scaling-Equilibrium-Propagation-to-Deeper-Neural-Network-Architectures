import argparse
import os
import torch
import wandb
from datasets import load_dataloaders
from model.hopfield.network import ConvHopfieldResEnergy32
from model.function.network import Network
from model.function.cost import CrossEntropy, SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer,adamOptimizer

parser = argparse.ArgumentParser(description='Best results obtained with a deep convolutional Hopfield network')
parser.add_argument('--dataset', type = str, default = 'FashionMNIST', help="The dataset used. Either `CIFAR10'  `CIFAR100' or `FashionMNIST'")
parser.add_argument('--epochs', type = int, default = 100, help="The number of epochs of training")

args = parser.parse_args()
if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark = True
    dataset = args.dataset
    use_warm_restarts=False 
    activation='relu6' #hard-sigmoid, relu6, relu, reluclip
    opt="sgd" #adam or sgd(nestrov momentum)
    lossfn="MSE" #CE for cross entropy, MSE for mean squared error

    # The best hyperparameters found, depending on the dataset (CIFAR10 or CIFAR100)
    if dataset == 'CIFAR10':
        num_outputs = 10
        nudging = 0.1 #0.1-0.4 works well

        # Number of inference/training steps
        num_iterations_inference = 120   
        num_iterations_training  = 50   
        weight_gains = [
    0.6, 0.6, 0.7,   # block-1 (skip stronger)
    0.6, 0.7, 0.6,   # block-2
    0.6, 0.7, 0.6,   # block-3
    0.6, 0.7, 0.6,   # block-4
    0.8              # dense head
]
        # Learning rates per weight 
        learning_rates_weights = [3e-2]*len(weight_gains)
        learning_rates_biases = learning_rates_weights.copy()

        momentum = 0.9
        weight_decay = 0e-4
        num_inputs = 1 #number of input channels
    elif dataset == 'CIFAR100':
        num_outputs = 100
        nudging = 0.15

        # Number of inference/training steps
        num_iterations_inference = 120   
        num_iterations_training  = 50   

        weight_gains = [
    0.6, 0.6, 0.7,   # block-1 (skip stronger)
    0.6, 0.7, 0.6,   # block-2
    0.6, 0.7, 0.6,   # block-3
    0.6, 0.7, 0.6,   # block-4
    0.8              # dense head
]

        learning_rates_weights = [3e-2]*len(weight_gains)
        learning_rates_biases = learning_rates_weights.copy()

        momentum = 0.9
        weight_decay = 3.5e-4
        num_inputs = 3 #number of input channels
    elif dataset == 'FashionMNIST':
        num_outputs = 10
        nudging = 0.15

        # Number of inference/training steps
        num_iterations_inference = 120   
        num_iterations_training  = 50  

        weight_gains = [
    0.6, 0.6, 0.7,   # block-1 (skip stronger)
    0.6, 0.7, 0.6,   # block-2
    0.6, 0.7, 0.6,   # block-3
    0.6, 0.7, 0.6,   # block-4
    0.8              # dense head
]

        learning_rates_weights = [3e-2]*len(weight_gains)
        learning_rates_biases = learning_rates_weights.copy()

        momentum = 0.9
        weight_decay = 3.5e-4
        num_inputs = 1 #number of input channels
    else:
        raise ValueError("expected 'CIFAR10' or 'CIFAR100' or FashionMNIST but got {}".format(dataset))


    batch_size = 128
    wandb.init(
        project="HopResnet CEP ",  
        name=f"{activation}_beta{nudging}_lr{learning_rates_weights[0]:.0e}",
        config={
            "dataset": dataset,
            "num_outputs": num_outputs,
            "beta": nudging,
            "num_iterations_inference": num_iterations_inference,
            "num_iterations_training": num_iterations_training,
            "weight_gains": weight_gains,
            "learning_rates_weights": learning_rates_weights,
            "learning_rates_biases": learning_rates_biases,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "activation": activation,
            "batch size": batch_size,
            "optimizer":opt,
            "loss fn":lossfn,
        }
    )

    # Load the training and test data (either CIFAR-10, CIFAR-100 or Fashion-MNIST)
    training_loader, test_loader = load_dataloaders(dataset, batch_size)
    

    if torch.cuda.is_available(): device = "cuda"
    else: device = "cpu"
    energy_fn = ConvHopfieldResEnergy32(num_inputs, num_outputs, weight_gains=weight_gains,activation=activation)

    # Set the device on which we run and train the network
    energy_fn.set_device(device)
    
    
    # Define the cost function: mean squared error (MSE)
    output_layer = energy_fn.layers()[-1]
    if lossfn=="CE":
        cost_fn=CrossEntropy(output_layer)
    else:
        cost_fn = SquaredError(output_layer)
    network = Network(energy_fn)

    # Define the energy minimizer used in the perturbed phase of training
    params = energy_fn.params()
    layers = energy_fn.layers()
    free_layers = network.free_layers()
    augmented_fn = AugmentedFunction(energy_fn, cost_fn)
    energy_minimizer_training = FixedPointMinimizer(augmented_fn, free_layers)
    energy_minimizer_training.num_iterations = num_iterations_training

    # Define the gradient estimator: centered equilibrium propagation (CEP)
    estimator = EquilibriumProp(params, layers, augmented_fn, cost_fn, energy_minimizer_training)
    estimator.variant = 'centered'
    estimator.nudging = nudging

    # Build the optimizer (SGD with momentum and weight decay)
    learning_rates = learning_rates_biases + learning_rates_weights
    if opt=="adam":
        optimizer = adamOptimizer(energy_fn, cost_fn, learning_rates,  weight_decay)
    else:
        optimizer = Optimizer(energy_fn, cost_fn, learning_rates, momentum, weight_decay)

    # Define the energy minimizer used at inference (fixed point minimizer)
    energy_minimizer_inference = FixedPointMinimizer(energy_fn, free_layers)
    energy_minimizer_inference.num_iterations = num_iterations_inference

    # Define the trainer (to perform one epoch of training) and the evaluator (to evaluate the model on the test set)
    trainer = Trainer(network, cost_fn, params, training_loader, estimator, optimizer, energy_minimizer_inference)
    evaluator = Evaluator(network, cost_fn, test_loader, energy_minimizer_inference)
    
    # Define the scheduler for the learning rates (cosine annealing scheduler)
    num_epochs = args.epochs
    if use_warm_restarts:                                  # ← if you let this be an arg/flag
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0    = 5,        # restart every 5 epochs
            T_mult = 1,
            eta_min= 2e-6,
        )
        sched_cfg = dict(
            type   = "CosineAnnealingWarmRestarts",
            T_0    = 5,
            T_mult = 1,
            eta_min= 2e-6,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max   = num_epochs,
            eta_min = 2e-6,
        )
        sched_cfg = dict(
            type   = "CosineAnnealingLR",
            T_max  = num_epochs,
            eta_min= 2e-6,
        )

    if wandb.run is not None:
        wandb.config.update({"scheduler": sched_cfg}, allow_val_change=True)


    # Define the path and the monitor to perform the run
    path = '/'.join(['save', dataset])
    if not os.path.exists(path): os.makedirs(path)
    monitor = Monitor(energy_fn, cost_fn, trainer, scheduler, evaluator, path)

    # Print the characteristics of the run
    print('Dataset: {} -- batch_size={}'.format(dataset, batch_size))
    print('Network: ', energy_fn)
    print('Cost function: ', cost_fn)
    print('Energy minimizer during inference: ', energy_minimizer_inference)
    print('Energy minimizer during training: ', energy_minimizer_training)
    print('Gradient estimator: ', estimator)
    print('Parameter optimizer: ', optimizer)
    print('Number of epochs = {}'.format(num_epochs))
    print('Path = {}'.format(path))
    print('Device = {}'.format(device))
    print()

    # Launch the experiment
    monitor.run(num_epochs, verbose=True)

    torch.save(energy_fn, 'model_full.pkl') 