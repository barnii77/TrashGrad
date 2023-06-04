import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #self.fc1 = nn.Dense(9216, 9216)
        self.conv1 = torch.nn.Conv2d(1, 4, (3, 3))
        self.conv2 = torch.nn.Conv2d(4, 1, (3, 3))
        #self.features = [self.conv1, self.conv2]#, self.fc1

    def forward(self, t):
        out = self.conv1(t)
        out = F.relu(out)
        out = self.conv2(out)
        out = F.sigmoid(out)
        #shape = out.shape
        #out = out.reshape((out.shape[0], -1))
        #print(shape[0], out.size)
        #out = self.fc1(out)
        #out = F.relu(out, .05)
        #out = out.reshape(shape)
        return out


x, y = torch.meshgrid(torch.linspace(-1, 1, 100), torch.linspace(-1, 1, 100))
x_data = torch.exp(-(x ** 2 + y ** 2) / 0.1).reshape(1, 1, 100, 100)
y_data = x_data[:, :, 2:-2, 2:-2]

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Train the model
num_epochs = 600
losses = []


def train():
    global y_pred
    for _ in range(num_epochs):
        #for epoch in range(num_epochs):
        y_pred = model(x_data)
        loss = torch.nn.functional.mse_loss(y_pred, y_data)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()


#cProfile.run("train()")
print(timeit(train, number=1))

# Plot the loss curve
plt.plot(np.log(np.array(losses)))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Plot the predicted and true output
y_pred_data = y_pred.detach().numpy().squeeze()

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(x_data.numpy().squeeze(), cmap="viridis", origin="lower")
axs[0].set_title("True output")
axs[1].imshow(y_pred_data, cmap="viridis", origin="lower")
axs[1].set_title("Predicted output")
plt.show()
