import argparse
import os
import torch
import wandb
from datasets import load_dataloaders
from model.hopfield.network import ConvHopfieldEnergy32, ConvHopfieldResEnergy32
from model.function.network import Network
from model.function.cost import CrossEntropy, SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.sgd import EquilibriumProp, AugmentedFunction
from training.epoch import Trainer, Evaluator
from training.monitor import Monitor, Optimizer,adamOptimizer

parser = argparse.ArgumentParser(description='Best results obtained with a deep convolutional Hopfield network')
parser.add_argument('--dataset', type = str, default = 'FashionMNIST',choices=['CIFAR10', 'CIFAR100','FashionMNIST'], help="The dataset used for training. Either 'CIFAR10' or 'CIFAR100' or 'FashionMNIST'")
parser.add_argument('--epochs', type = int, default = 100, help="The number of epochs of training")
parser.add_argument('--activation', type = str, default = 'relu6', choices=['hard-sigmoid', 'relu6', 'relu', 'reluclip'], help="The activation function. Either 'hard-sigmoid', 'relu6', 'relu' or 'reluclip'")
parser.add_argument('--lossfn', type = str, default = 'MSE', choices=['MSE', 'CE'], help="The loss function. Either 'MSE' (mean squared error) or 'CE' (cross-entropy)")
parser.add_argument('--optimizer', type = str, default = 'sgd',choices=['sgd', 'adam'], help="The optimizer. Either 'sgd' (NAG) or 'adam' (with adam use learning rate 1e-4)")
parser.add_argument('--use_warm_restarts', action='store_true',default=False, help="If set, use cosine annealing with warm restarts as a scheduler for the learning rates else use standard cosine annealing")
parser.add_argument('--beta', type = float, default = 0.1, help="The nudging factor best found CIFAR10=0.1, CIFAR100=0.15, FashionMNIST=0.15")
parser.add_argument('--num_iterations_inference', type = int, default = 120, help="The number of inference iterations ")
parser.add_argument('--num_iterations_training', type = int, default = 50, help="The number of training iterations")
parser.add_argument('--weight_gains', type = str, default = None, help="The weight gains, as a comma-separated list of floats, one per weight matrix (in total 13 for HopfieldResNet13, 5 for VGG5)")
parser.add_argument('--learning_rates_weights', type = str, default = None, help="The learning rates for the weights, as a comma-separated list of floats, one per weight matrix ")
parser.add_argument('--learning_rates_biases', type = str, default = None, help="The learning rates for the biases, as a comma-separated list of floats, one per weight matrix")
parser.add_argument('--momentum', type = float, default = 0.9, help="The momentum factor for the SGD optimizer best found 0.9")
parser.add_argument('--weight_decay', type = float, default = 3.5e-4, help="The weight decay factor for the SGD optimizer best found CIFAR10=2.5e-4, CIFAR100=3.5e-4, FashionMNIST=3.5e-4")
parser.add_argument('--batch_size', type = int, default = 128, help="The batch size")
parser.add_argument('--network',type=str, default='hopfieldresnet13',choices=['hopfieldresnet13', 'vgg5'], help="The network architecture. Either `hopfieldresnet13'  or `vgg5'")
parser.add_argument('--no_wandb', action='store_true', default=False, help="If set, do not use wandb to log the results only print in console")
args = parser.parse_args()
if __name__ == "__main__":
    
    torch.backends.cudnn.benchmark = True
    dataset = args.dataset
    use_warm_restarts=False 
    activation=args.activation  #hard-sigmoid, relu6, relu, reluclip
    opt=args.optimizer #sgd or adam
    lossfn= args.lossfn #MSE or CE
    momentum = args.momentum
    weight_decay = args.weight_decay
    nudging = args.beta
    num_iterations_inference = args.num_iterations_inference
    num_iterations_training = args.num_iterations_training
    network = args.network #hopfieldresnet13 or vgg5
    if args.weight_gains is not None:
        weight_gains = [float(w) for w in args.weight_gains.split(',')]
    else:
        if network=='hopfieldresnet13':
                    weight_gains = [
                    0.6, 0.6, 0.7,   # block-1 (skip stronger)
                    0.6, 0.7, 0.6,   # block-2
                    0.6, 0.7, 0.6,   # block-3
                    0.6, 0.7, 0.6,   # block-4
                    0.8              # dense head
                ]        
        elif network=='vgg5':
            weight_gains = [.4, .7, .6, .3, .6]
        else:
            raise ValueError("expected 'hopfieldresnet13' or 'vgg5' but got {}".format(network))
    if args.learning_rates_weights is not None:
        learning_rates_weights = [float(lr) for lr in args.learning_rates_weights.split(',')]
    else:
        if opt=='adam':
            learning_rates_weights = [1e-4]*len(weight_gains)
        else:
            learning_rates_weights = [3e-2]*len(weight_gains)
    if args.learning_rates_biases is not None:
        learning_rates_biases = [float(lr) for lr in args.learning_rates_biases.split(',')]
    else:
        learning_rates_biases = learning_rates_weights.copy()
    # The best hyperparameters found, depending on the dataset (CIFAR10 or CIFAR100)
    if dataset == 'CIFAR10':
        num_outputs = 10
        num_inputs = 3 #number of input channels
    elif dataset == 'CIFAR100':
        num_outputs = 100
        
        num_inputs = 3 #number of input channels
    elif dataset == 'FashionMNIST':
        num_outputs = 10
        num_inputs = 1 #number of input channels
    else:
        raise ValueError("expected 'CIFAR10' or 'CIFAR100' or FashionMNIST but got {}".format(dataset))
    batch_size = args.batch_size
    if not args.no_wandb:
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
    if network=='hopfieldresnet13':
        energy_fn = ConvHopfieldResEnergy32(num_inputs, num_outputs, weight_gains=weight_gains,activation=activation)
    elif network=='vgg5':
        energy_fn = ConvHopfieldEnergy32(num_inputs, num_outputs, weight_gains=weight_gains,activation=activation)

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