import matplotlib.pyplot as plt
import torch.optim as optim
import torch
from torch import nn
from torchvision.io import read_image
import torch.nn.functional as F
from tqdm.auto import tqdm


#
# min int_0^T int_{\Omega} (m1**2 + m2**2)/rho
# s.t Dx m1 + Dy m2 + Dt rho = 0
#     rho(x,y,0) = I1    rho(x,y,1) = I2
# Use staggered grid for m1, m2, rho
# Solve the constraint optimization using exact penalty function
#

class omtFD(nn.Module):  # X,Y,T version
    def __init__(self, n1, n2, n3, bs=1):
        super().__init__()
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.bs = bs

        self.m1 = nn.Parameter(torch.zeros(bs, 1, n1 - 1, n2, n3))
        self.m2 = nn.Parameter(torch.zeros(bs, 1, n1, n2 - 1, n3))
        self.rho = nn.Parameter(torch.zeros(bs, 1, n1, n2, n3 - 1) - 3)

    def OMTconstraintMF(self, I1, I2):
        dx, dy, dt = 1 / self.n1, 1 / self.n2, 1 / self.n3
        # bs = I1.shape[0]

        ox = torch.zeros(self.bs, 1, 1, self.n2, self.n3).to(self.rho.device)
        oy = torch.zeros(self.bs, 1, self.n1, 1, self.n3).to(self.rho.device)

        I1 = I1.unsqueeze(-1)
        I2 = I2.unsqueeze(-1)
        rhop = torch.exp(self.rho)

        # Add I1 and I2 to rho
        rhoBC = torch.cat((I1, rhop, I2), dim=-1)
        m1BC = torch.cat((ox, self.m1, ox), dim=-3)
        m2BC = torch.cat((oy, self.m2, oy), dim=-2)

        # Compute the Div
        m1x = (m1BC[:, :, :-1, :, :] - m1BC[:, :, 1:, :, :]) / dx
        m2y = (m2BC[:, :, :, :-1, :] - m2BC[:, :, :, 1:, :]) / dy
        rhot = (rhoBC[:, :, :, :, :-1] - rhoBC[:, :, :, :, 1:]) / dt
        return m1x + m2y + rhot

    def OMTgradRho(self, I1, I2):
        I1 = I1.unsqueeze(-1)
        I2 = I2.unsqueeze(-1)

        # Add I1 and I2 to rho
        rhop = torch.exp(self.rho)
        rhoBC = torch.cat((I1, rhop, I2), dim=-1)
        rhox = (rhoBC[:, :, :-1, :, :] - rhoBC[:, :, 1:, :, :])
        rhoy = (rhoBC[:, :, :, :-1, :] - rhoBC[:, :, :, 1:, :])
        rhot = (rhoBC[:, :, :, :, :-1] - rhoBC[:, :, :, :, 1:])

        return rhox, rhoy, rhot

    def OMTobjfunMF(self, I1, I2):
        I1 = I1.unsqueeze(-1)
        I2 = I2.unsqueeze(-1)

        rhop = torch.exp(self.rho)

        ox = torch.zeros(self.bs, 1, 1, self.n2, self.n3).to(self.rho.device)
        oy = torch.zeros(self.bs, 1, self.n1, 1, self.n3).to(self.rho.device)

        # Add I1 and I2 to rho
        rhoBC = torch.cat((I1, rhop, I2), dim=-1)
        m1BC = torch.cat((ox, self.m1, ox), dim=-3)
        m2BC = torch.cat((oy, self.m2, oy), dim=-2)

        # Average quantities to cell center
        m1c = (m1BC[:, :, :-1, :, :] + m1BC[:, :, 1:, :, :]) / 2
        m2c = (m2BC[:, :, :, :-1, :] + m2BC[:, :, :, 1:, :]) / 2
        rhoc = (rhoBC[:, :, :, :, :-1] + rhoBC[:, :, :, :, 1:]) / 2

        f = ((m1c ** 2 + m2c ** 2) / rhoc).mean()
        return f


def LearnOMT(x0, x1, nt=20, device='cpu'):
    # initilize images
    # x0.shape :        torch.Size([ 256, 2, 64, 64])  MovingMNIST  bs,ch,x,y

    t = torch.linspace(0, 1, nt)

    eps = 1e-2#4

    I1_min = x0.min()
    I2_min = x1.min()
    I1_mean = (x0 - I1_min + eps).mean()
    I2_mean = (x1 - I2_min + eps).mean()

    I1 = (x0 - I1_min + eps) / I1_mean
    I2 = (x1 - I2_min + eps) / I2_mean

    bs = I1.shape[0]
    ch = I1.shape[1]
    n1 = I1.shape[2]
    n2 = I1.shape[3]
    n3 = nt

    omt = omtFD(n1, n2, n3, bs=bs * ch).to(device)

    # input shape for omt: (bs, 1, nx, ny)
    # final output shape for omt: (bs, 1, nx, ny, nt)

    I1 = I1.reshape([bs * ch, 1, n1, n2])
    I2 = I2.reshape([bs * ch, 1, n1, n2])

    optimizer = optim.Adam(omt.parameters(), lr=1e-2)

    num_epochs = 1000  # with incomplete optimization you do not see the transport, rather it appears as one dims and the other brightens up
    opt_iter = 20
    torch.autograd.set_detect_anomaly(True)

    # Train the model
    mu = 1e-2
    delta = 1e-1
    # initialize lagrange multiplier
    p = torch.zeros(n1, n2, n3).to(device)
    for epoch in range(num_epochs):
        # Optimization iterations for fixed p
        for jj in range(opt_iter):
            # Zero the parameter gradients
            optimizer.zero_grad()

            # function evaluation
            f = omt.OMTobjfunMF(I1, I2)
            # Constraint
            c = omt.OMTconstraintMF(I1, I2)

            const = -(p * c).mean() + 0.5 * mu * (c ** 2).mean()

            loss = f + const

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_value_(omt.parameters(), clip_value=0.5)
            optimizer.step()

        print('f:', f.item(), '\tconst:', const.item())
        # Update Lagrange multiplier
        with torch.no_grad():
            p = p - delta * mu * c

    rho = omt.rho.detach()
    rhop = torch.exp(rho)
    # Add I1 and I2 to rho
    rhop = torch.cat((I1.unsqueeze(-1), rhop, I2.unsqueeze(-1)), dim=-1)

    rhop = rhop.reshape([bs, ch, n1, n2, n3 + 1])

    I2_mean = (x1 - I2_min + eps).mean()

    rhop = rhop * I1_mean - eps + I1_min  # assuming mass of x0 = mass of x1

    rhop = rhop.detach()
    # shape of rhop: (bs, ch, nx, ny, nt+1) includes x0, "nt-1" timesteps, x1

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))
    axs[0, 0].imshow((rhop[0, 0, :, :, 0]).detach().cpu().numpy())
    axs[0, 0].set_title('0')
    axs[0, 1].imshow((rhop[0, 0, :, :, 1]).detach().cpu().numpy())
    axs[0, 1].set_title('1')
    axs[0, 2].imshow((rhop[0, 0, :, :, 5]).detach().cpu().numpy())
    axs[0, 2].set_title('5')
    # axs[1, 0].imshow((rhop[0, 0, :, :, 10]).detach().cpu().numpy())
    # axs[1, 0].set_title('10')
    # axs[1, 1].imshow((rhop[0, 0, :, :, 15]).detach().cpu().numpy())
    # axs[1, 1].set_title('15')
    # axs[1, 2].imshow((rhop[0, 0, :, :, 20]).detach().cpu().numpy())
    # axs[1, 2].set_title('20')
    axs[2, 0].imshow((rhop[0, 0, :, :, -5]).detach().cpu().numpy())
    axs[2, 0].set_title('5th Last')
    axs[2, 1].imshow((rhop[0, 0, :, :, -2]).detach().cpu().numpy())
    axs[2, 1].set_title('2nd Last')
    axs[2, 2].imshow((rhop[0, 0, :, :, -1]).detach().cpu().numpy())
    axs[2, 2].set_title('Last')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig("plots/OT.png")
    # plt.show()
    print('saved plot')

    return rhop


device = 'cuda:0'
x0 = read_image('HealthyCell2.png')[0]
x1 = read_image('OldCell1.png')[0]
# shape of x0 and x1 = [:,:]
size_img = 64
#breakpoint()
xt = torch.load('xt.pt', map_location='cpu')  # [128, 10, 64, 64]
t = torch.load('t.pt', map_location='cpu')
# t[6] is 0
#plt.imshow(xt[6,0,:,:].detach().numpy())
x0 = xt[6,0,:,:]
x1 = xt[6,7,:,:]
torch.save(x0,'x0.pt')
torch.save(x1,'x1.pt')
x0 = F.interpolate(x0.unsqueeze(0).unsqueeze(0), size = [size_img,size_img])#/255.
x1 = F.interpolate(x1.unsqueeze(0).unsqueeze(0), size = [size_img,size_img])#/255.
#breakpoint()
rhop = LearnOMT(x0.to(device), x1.to(device), nt=10, device='cuda:0')
rhop = rhop[0,0].detach().cpu().numpy()

x0 = x0[0,0].unsqueeze(-1)
x1 = x1[0,0].unsqueeze(-1)
Frac = torch.linspace(0,1,rhop.shape[-1])
Frac = Frac.unsqueeze(0).unsqueeze(0)
LinearInt = x1*Frac + (1-Frac)*x0
LinearInt = LinearInt.detach().cpu().numpy()

fig, ax = plt.subplots(nrows=2, ncols=rhop.shape[-1], figsize=(50, 5))

for i in range(rhop.shape[-1]):
    ax[0, i].imshow(LinearInt[:, :, i])
    ax[0, i].axis('off')
    ax[1, i].imshow(rhop[:, :, i])
    ax[1, i].axis('off')


ax[0,0].set_title('Linear Interpolation', loc='left')
ax[1,0].set_title('OMT Interpolation', loc='left')
plt.savefig('plots/comparison.png')
plt.show()
breakpoint()

fig, ax = plt.subplots(nrows=2, ncols=rhop.shape[-1], figsize=(100, 20))

for i in range(rhop.shape[-1]):
    ax[0, i].imshow(rhop[:, :, i])
    ax[0, i].axis('off')
    ax[1, i].imshow(LinearInt[:, :, i])
    ax[1, i].axis('off')

plt.savefig('plots/comparison2.png')
plt.show()