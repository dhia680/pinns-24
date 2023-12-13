from PIL import Image

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import tqdm


def save_gif_PIL(outfile, files, fps=5, loop=0):
    """Helper function for saving GIFs"""
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=int(1000 / fps), loop=loop)


class Oscillator:
    """
    Equation
        m y''(x) + mu y'(x) + k y(x) = 0,  x in (0, 1)
    Initial condition
        y(0) = 1
        y'(0) = 1
    """
    def __init__(self, d, w0):
        assert d < w0
        self.d = d
        self.w0 = w0

    def exact_solution(self, x):
        """Defines the analytical solution to the 1D underdamped harmonic
        oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/"""
        w = np.sqrt(self.w0 ** 2 - self.d ** 2)
        phi = np.arctan(-self.d / w)
        A = 1 / (2 * np.cos(phi))
        cos = torch.cos(phi + w * x)
        sin = torch.sin(phi + w * x)
        exp = torch.exp(-self.d * x)
        return exp * 2 * A * cos

    def residual(self, dy_dxx, dy_dx, y):
        mu = 2 * self.d
        k = self.w0 ** 2
        return dy_dxx + mu * dy_dx + k * y

    def fd_residual(self, x, y):
        x = x.reshape(-1)[1:-1]
        y = y.reshape(-1)[1:-1]
        dy_dx = np.gradient(y, x)
        dy_dxx = np.gradient(dy_dx, x)
        return self.residual(dy_dxx, dy_dx, y)


class SinActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class FullyConnectedNetwork(nn.Module):
    """Defines a connected network"""

    def __init__(self, input_dim, output_dim, hidden_dim, nb_layers):
        super().__init__()
        activation = nn.Tanh
        # activations = [nn.Tanh for _ in range(nb_layers - 1)]
        activations = [SinActivation for _ in range(nb_layers - 1)]
        activations[-1] = SinActivation
        self.fcs = nn.Sequential(nn.Linear(input_dim, hidden_dim), activation())
        self.fch = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act())
              for act in activations])
        self.fce = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x


def plot_result(x_exact, y_exact, xp, yh, epoch, file_name):
    """Pretty plot training results"""
    plt.figure(figsize=(8, 4))
    plt.plot(x_exact, y_exact, color="grey", linewidth=2,
             alpha=0.8,
             label="Exact solution")
    plt.plot(x_exact, yh, color="tab:blue", linewidth=4, alpha=0.8,
             label="Neural network prediction")
    plt.scatter(xp, -0 * torch.ones_like(xp), s=60, color="tab:green",
                alpha=0.4,
                label='Physics loss training locations')
    l = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(l.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065, 0.7, "Training step: %i" % (epoch + 1), fontsize="xx-large",
             color="k")
    plt.axis("off")
    plt.savefig(file_name, bbox_inches='tight',
                pad_inches=0.1, dpi=100, facecolor="white")


def plot_learning(epoch_list,
                  loss_residual_list,
                  loss_y0_list,
                  loss_dy_dx_0_list,
                  total_loss_list):
    fig = plt.figure()
    ax = fig.subplots(2, 2)
    ax[0][0].semilogy(epoch_list, total_loss_list, label="Total loss")
    ax[0][1].semilogy(epoch_list, loss_dy_dx_0_list, label="loss dy/dx[0]**2")
    ax[1][0].semilogy(epoch_list, loss_y0_list, label="loss (y[0] - 1)**2")
    ax[1][1].semilogy(epoch_list, loss_residual_list, label="loss residual")
    [(a.grid(), a.legend()) for aa in ax for a in aa]
    fig.savefig("loss_convergence")


def train_from_data(oscillator: Oscillator):
    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # generate data
    # get the analytical solution over the full domain
    nx = 500
    x = torch.linspace(0, 1, nx).view(-1, 1)
    y = oscillator.exact_solution(x).view(-1, 1)

    # slice out a small number of points from the LHS of the domain
    n_data = 50
    idx = [int(z) for z in np.linspace(0, nx - 1, n_data)]
    x_data = x[idx]
    y_data = y[idx]

    # plot exact solution and the data
    plt.figure()
    plt.plot(x, y, label="Exact solution")
    plt.scatter(x_data, y_data, color="tab:orange", label="Training data")
    plt.legend()
    plt.savefig("exact_sol_and_data")

    # train standard neural network to fit training data
    model = FullyConnectedNetwork(input_dim=1, output_dim=1,
                                  hidden_dim=32,
                                  nb_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    files = []
    n_epoch = 10_000
    for epoch in tqdm.tqdm(range(n_epoch)):
        optimizer.zero_grad()
        yh = model(x_data)
        loss = torch.mean((yh - y_data) ** 2)  # use mean squared error
        loss.backward()
        optimizer.step()

        # plot the result as training progresses
        if (epoch + 1) % (n_epoch // 100) == 0:

            yh = model(x).detach()

            file_name = f"{plots_dir}/nn_{epoch + 1:8i}.png"
            plot_result(x_exact=x, y_exact=y,
                        xp=x_data, yh=yh,
                        epoch=epoch, file_name=file_name)

            files.append(file_name)

    save_gif_PIL("nn.gif", files, fps=20, loop=0)


def train_pinns(oscillator: Oscillator):
    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # get the analytical solution over the full domain
    n_pts = 500
    x_exact = torch.linspace(0, 1, n_pts).view(-1, 1).requires_grad_(True)
    y_exact = oscillator.exact_solution(x_exact).view(-1, 1)

    # sample locations over the problem domain
    n_col_pts = 100
    collocation_pts = torch.linspace(0, 1,
                                     n_col_pts,
                                     requires_grad=True).view(-1, 1)

    model = FullyConnectedNetwork(input_dim=1,
                                  output_dim=1,
                                  hidden_dim=32,
                                  nb_layers=3)

    learning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    n_epoch = 50_000

    def lr_multiplier(epoch):
        return 0.1 ** (epoch / n_epoch)

    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lr_lambda=lr_multiplier)
    files = []
    epoch_list = []
    loss_residual_list = []
    loss_y0_list = []
    loss_dy_dx_0_list = []
    total_loss_list = []
    for epoch in tqdm.tqdm(range(n_epoch)):
        optimizer.zero_grad()
        y = model(collocation_pts)
        # computes dy/dx
        dy_dx = torch.autograd.grad(y, collocation_pts,
                                    torch.ones_like(y),
                                    create_graph=True)[0]

        # computes d^2y/dx^2
        dy_dxx = torch.autograd.grad(dy_dx, collocation_pts,
                                     torch.ones_like(dy_dx),
                                     create_graph=True)[0]

        # computes the residual of the 1D harmonic oscillator differential
        residual = oscillator.residual(dy_dxx, dy_dx, y)
        loss_residual = torch.mean(residual ** 2)
        loss_y0 = (y[0] - 1.0) ** 2
        loss_dy_dx_0 = dy_dx[0] ** 2

        # backpropagate joint loss
        alpha = 1e-6
        loss = alpha * loss_residual + loss_y0 + loss_dy_dx_0

        # terms together
        loss.backward()
        optimizer.step()
        scheduler.step()

        # plot the result as training progresses
        if (epoch + 1) % 300 == 0:
            print("")
            print(f"Learning rate = {optimizer.param_groups[0]['lr']:.2e}")
            print(f"Losses")
            print(f"    - Residual loss     = "
                  f"{loss_residual.item():.2e}")
            print(f"    - Initial cond loss = "
                  f"{loss_y0.item():.2e}   {loss_dy_dx_0.item():.2e}")
            print(f"=> Total weighted loss = {loss.item():.2e}")
            yh = model(x_exact).detach()
            xp = collocation_pts.detach()

            file_name = "plots/pinn_%.8i.png" % (epoch + 1)
            plot_result(x_exact=x_exact.detach().numpy(),
                        y_exact=y_exact.detach().numpy(),
                        xp=xp, yh=yh, epoch=epoch,
                        file_name=file_name)
            files.append(file_name)

            epoch_list.append(epoch)
            loss_residual_list.append(loss_residual.item())
            loss_y0_list.append(loss_y0.item())
            loss_dy_dx_0_list.append(loss_dy_dx_0.item())
            total_loss_list.append(loss.item())

    save_gif_PIL("pinn.gif", files, fps=20, loop=0)
    plot_learning(epoch_list,
                  loss_residual_list,
                  loss_y0_list,
                  loss_dy_dx_0_list,
                  total_loss_list)


if __name__ == '__main__':
    torch.manual_seed(123)
    oscillator = Oscillator(d=2, w0=20)
    compute_pinn = True
    compute_from_data = False
    if compute_pinn:
        train_pinns(oscillator)
    if compute_from_data:
        train_from_data(oscillator)
    print("done!")
