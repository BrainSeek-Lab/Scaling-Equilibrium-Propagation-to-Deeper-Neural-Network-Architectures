import torch
import matplotlib.pyplot as plt

from datasets import load_dataloaders
from model.hopfield.network import ConvHopfieldEnergy32, ConvHopfieldResEnergy32
from model.function.network import Network
from model.function.cost import SquaredError
from model.hopfield.minimizer import FixedPointMinimizer
from training.epoch import  Evaluator
from training.monitor import Monitor



def run_evaluation_only(
    model_path,
    test_loader,
    num_iterations,
    device='cuda',
    num_inputs=3,
    num_outputs=10,
    activation='relu6',
    weight_gains=None
):
    if weight_gains is None:
        weight_gains = [
            0.4, 0.7, 0.6, 0.5,
            0.3, 0.4, 0.4, 0.4,
            0.5
        ]
    # Try loading the model
    loaded_obj = torch.load(model_path, map_location=device)

    if isinstance(loaded_obj, ConvHopfieldEnergy32): #same as class name, can be ConvHopfieldResEnergy32 as well
        # Case 1: Loaded full model from pickle
        energy_fn = loaded_obj
        energy_fn.set_device(device)


    else:
        # Case 2: Only weights are loaded(pt), rebuild model
        energy_fn = ConvHopfieldEnergy32( #ConvHopfieldResEnergy32
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            activation=activation,
            weight_gains=weight_gains
        )
        energy_fn.set_device(device)
        energy_fn.load(model_path)
    for i, param in enumerate(energy_fn._params):
        values = param.get().detach().cpu().numpy().ravel()  # flatten

        # file-safe name
        pname = getattr(param, "name", f"param_{i}")
        pshape = tuple(param.get().shape)

        # create a new figure per parameter
        plt.figure(figsize=(8, 6))
        plt.hist(values, bins=100, alpha=0.7, density=True)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.title(f"Distribution of {pname} {pshape}")
        plt.tight_layout()

        # save with unique filename
        # plt.savefig(f"{pname}_distribution.png", dpi=150, bbox_inches="tight")
        plt.show()
        # plt.close()  # close to avoid memory buildup

    # Evaluation pipeline
    print(energy_fn)
    cost_fn = SquaredError(energy_fn.layers()[-1])
    network = Network(energy_fn)
    free_layers = network.free_layers()

    inference_minimizer = FixedPointMinimizer(energy_fn, free_layers)
    inference_minimizer.num_iterations = num_iterations

    evaluator = Evaluator(network, cost_fn, test_loader, inference_minimizer)
    monitor = Monitor(energy_fn, cost_fn, None, None, evaluator, path="eval_only_logs")

    monitor.evaluate_only(verbose=True)


if __name__ == "__main__":
    num_outputs = 10
    nudging = 0.1
    num_iterations_inference =100
    num_iterations_training = 40
    dataset = 'CIFAR10'
    batch_size = 128
    training_loader, test_loader = load_dataloaders(dataset, batch_size)
    run_evaluation_only('save/CIFAR10/model.pt', test_loader, num_iterations_inference, device='cuda')
    # run_evaluation_only('model.pt', test_loader, num_iterations_inference, device='cuda') #loaing from pythorch weights as well as pickle objects is supported if from weight set hyperparams are as the training 