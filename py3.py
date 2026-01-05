import torch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs): #A
        super().__init__()
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30), #B
            torch.nn.ReLU(), #C
            # 2nd hidden layer
            torch.nn.Linear(30, 20), #D
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )
    def forward(self, x):
        logits = self.layers(x)
        return logits #E
    
model = NeuralNetwork(50, 3)
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)

print(model.layers[0].weight)
print(model.layers[0].weight.shape)