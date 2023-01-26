from inspect import signature
import numpy as np
import torch
from torch import tensor
import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.models.signature import infer_signature


class LinearNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)  # One in and one out

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def gen_data():
    # Example linear model modified to use y = 2x
    # from https://github.com/hunkim/PyTorchZeroToAll
    # X training data, y labels
    a = np.arange(1.0, 25.0)
    b = np.array([x * 2 for x in a])
    # Then, convert to tensors
    X = torch.from_numpy(a).view(-1, 1)
    y = torch.from_numpy(b).view(-1, 1)
    assert type(X) == type(y)
    assert len(X) == len(y)
    X = X.to(torch.float32)
    y = y.to(torch.float32)
    return X, y


# Define model, loss, and optimizer
model = LinearNNModel()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# Training loop
epochs = 250
X, y = gen_data()
for epoch in range(epochs):
    # Forward pass: Compute predicted y by passing X to the model
    y_pred = model(X)

    # Compute the loss
    loss = criterion(y_pred, y)

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Test model predictions
new_X = tensor([[5.0]])
y_pred = model(new_X)
assert type(new_X) == type(y_pred)
# y_pred_2 = scripted_pytorch_model(new_X)
print(f"For x feature: {new_X} and the base model, the prediction is {y_pred}")
# print(f"For x feature: {new_X} and the scripted model, the prediction is {y_pred_2}")

signature_for_base = infer_signature(new_X.numpy(), y_pred.detach().numpy())
# signature_for_scripted = infer_signature(new_X, y_pred_2)


# Log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "model", signature=signature_for_base)

    # # convert to scripted model and log the model
    # scripted_pytorch_model = torch.jit.script(model)
    # mlflow.pytorch.log_model(scripted_pytorch_model, "scripted_model")

# Fetch the logged model artifacts
print("run_id: {}".format(run.info.run_id))
for artifact_path in ["model/data", "scripted_model/data"]:
    artifacts = [f.path for f in MlflowClient().list_artifacts(run.info.run_id, artifact_path)]
    print("artifacts: {}".format(artifacts))
