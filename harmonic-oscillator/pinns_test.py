from PIL import Image

import os
from collections import namedtuple
from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import tqdm


def save_gif_pil(outfile, files, fps=5, loop=0):
    """Helper function for saving GIFs"""
    imgs = [Image.open(file) for file in files]
    imgs[0].save(fp=outfile, format='GIF', append_images=imgs[1:],
                 save_all=True, duration=int(1000 / fps), loop=loop)


class CauchyProblem(ABC):
    @abstractmethod
    def reference_solution(self, x):
        pass

    @abstractmethod
    def residual(self, ddy, dy, y):
        pass

    @abstractmethod
    def cauchy_conditions(self, dy, y):
        pass


class Oscillator(CauchyProblem):
    """
    Equation
        m y''(x) + mu y'(x) + k y(x) = 0,  x in (0, 1)
    Initial condition
        y(0) = 1
        y'(0) = 1
    Here
        - dumping = mu / 2m
        - pulsation = sqrt(k/m)
    """
    def __init__(self, dumping, pulsation):
        assert dumping < pulsation
        self.d = dumping
        self.w0 = pulsation
        self.y0 = 1.0
        self.dy0 = 0.0

    def reference_solution(self, x):
        """Defines the analytical solution to the 1D underdamped harmonic
        oscillator problem.
        Equations taken from: https://beltoforion.de/en/harmonic_oscillator/
        """
        w = np.sqrt(self.w0 ** 2 - self.d ** 2)
        phi = np.arctan(-self.d / w)
        a = 1 / (2 * np.cos(phi))
        cos = torch.cos(phi + w * x)
        # sin = torch.sin(phi + w * x)
        exp = torch.exp(-self.d * x)
        return exp * 2 * a * cos

    def residual(self, ddy, dy, y):
        mu = 2 * self.d
        k = self.w0 ** 2
        return ddy + mu * dy + k * y

    def cauchy_conditions(self, dy, y):
        return namedtuple("CauchyCond", ["y0", "dy0"])(y[0] - self.y0,
                                                       dy[0] - self.dy0)


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
    legend = plt.legend(loc=(1.01, 0.34), frameon=False, fontsize="large")
    plt.setp(legend.get_texts(), color="k")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-1.1, 1.1)
    plt.text(1.065, 0.7, "Training step: %i" % (epoch + 1), fontsize="xx-large",
             color="k")
    plt.axis("off")
    plt.savefig(file_name, bbox_inches='tight',
                pad_inches=0.1, dpi=100, facecolor="white")


def plot_losses(losses):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.subplots(2, 2)
    epoch_list = range(len(losses.residual))
    ax[0][0].semilogy(epoch_list, losses.total,
                      color="black",
                      label="Total loss")
    ax[0][1].semilogy(epoch_list, losses.residual,
                      color="red",
                      label="loss residual")
    ax[1][0].semilogy(epoch_list, losses.init_value,
                      color="green",
                      label="loss (y[0] - 1)**2")
    ax[1][1].semilogy(epoch_list, losses.init_derivative,
                      color="green",
                      label="loss dy/dx[0]**2")
    [(a.grid(), a.legend()) for aa in ax for a in aa]
    fig.savefig("loss_convergence")


def train_from_data(cauchy_problem: CauchyProblem):
    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # generate data
    # get the analytical solution over the full domain
    nx = 500
    x = torch.linspace(0, 1, nx).view(-1, 1)
    y = cauchy_problem.reference_solution(x).view(-1, 1)

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

    save_gif_pil("nn.gif", files, fps=20, loop=0)


class SavedLosses:
    def __init__(self):
        self.total = []
        self.residual = []
        self.init_value = []
        self.init_derivative = []

    def save(self, total, residual, init_value, init_derivative):
        self.total.append(total)
        self.residual.append(residual)
        self.init_value.append(init_value)
        self.init_derivative.append(init_derivative)


def train_pinns(cauchy_problem: CauchyProblem):
    plots_dir = "./plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    # get the analytical solution over the full domain
    n_pts = 500
    x_exact = torch.linspace(0, 1, n_pts).view(-1, 1).requires_grad_(True)
    y_exact = cauchy_problem.reference_solution(x_exact).view(-1, 1)

    # sample locations over the problem domain
    n_col_pts = 30
    collocation_pts = torch.linspace(0, 1,
                                     n_col_pts,
                                     requires_grad=True).view(-1, 1)

    model = FullyConnectedNetwork(input_dim=1,
                                  output_dim=1,
                                  hidden_dim=32,
                                  nb_layers=3)

    coef_loss_residual = 1e-5
    coef_loss_y0 = 1e1
    coef_loss_dy0 = 1e-4

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate)

    n_epoch = 40_000

    def lr_multiplier(epoch):
        return 1 ** (epoch / n_epoch)

    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lr_lambda=lr_multiplier)
    files = []
    losses = SavedLosses()
    for epoch in tqdm.tqdm(range(n_epoch)):
        optimizer.zero_grad()
        y = model(collocation_pts)
        # computes dy/dx
        dy = torch.autograd.grad(y, collocation_pts,
                                 torch.ones_like(y),
                                 create_graph=True)[0]

        # computes d^2y/dx^2
        ddy = torch.autograd.grad(dy, collocation_pts,
                                  torch.ones_like(dy),
                                  create_graph=True)[0]

        # computes the residual of the 1D harmonic oscillator differential
        residual = cauchy_problem.residual(ddy, dy, y)
        loss_residual = torch.mean(residual ** 2)
        cauchy_cond = cauchy_problem.cauchy_conditions(dy, y)
        loss_y0, loss_dy0 = cauchy_cond.y0 ** 2, cauchy_cond.dy0 ** 2

        # backpropagate joint loss
        loss = (coef_loss_residual * loss_residual
                + coef_loss_y0 * loss_y0
                + coef_loss_dy0 * loss_dy0)

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.save(total=loss.item(),
                    residual=loss_residual.item(),
                    init_value=loss_y0.item(),
                    init_derivative=loss_dy0.item())

        # plot the result as training progresses
        if (epoch + 1) % 300 == 0:
            print("")
            print(f"* Learning rate = {optimizer.param_groups[0]['lr']:.2e}")
            print(f"* Losses")
            print(f"    - Residual loss     = "
                  f"{loss_residual.item():.2e}")
            print(f"    - Initial cond loss = "
                  f"{loss_y0.item():.2e}   {loss_dy0.item():.2e}")
            print(f"=> Total weighted loss = {loss.item():.2e}")
            yh = model(x_exact).detach()
            xp = collocation_pts.detach()

            file_name = f"{plots_dir}/pinn_{epoch + 1:8d}.png"
            plot_result(x_exact=x_exact.detach().numpy(),
                        y_exact=y_exact.detach().numpy(),
                        xp=xp, yh=yh, epoch=epoch,
                        file_name=file_name)
            files.append(file_name)

    save_gif_pil("pinn.gif", files, fps=20, loop=0)
    plot_losses(losses)


if __name__ == '__main__':
    torch.manual_seed(123)
    oscillator = Oscillator(dumping=2, pulsation=20)
    compute_pinn = True
    compute_from_data = False
    if compute_pinn:
        train_pinns(oscillator)
    if compute_from_data:
        train_from_data(oscillator)
    print("done!")
