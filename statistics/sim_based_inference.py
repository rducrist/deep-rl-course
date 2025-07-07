import torch
import sbi.utils as utils
from sbi.inference import infer

prior = utils.BoxUniform(
    low=torch.tensor([-5., -5.]),
    high=torch.tensor([5., 5.])
)

def simulator(mu):
    #generate samples from N(nu, sigma=0.5)
    return mu + 0.5 * torch.rand_like(mu)

# record 200 sim and fit the joint surface mu1, mu2, x1, x2 with a NN 
num_sim = 200
method = 'SNRE' #SNPE or SNLE or SNRE
posterior = infer(
    simulator,
    prior,
    # See glossary for explanation of methods.
    #    SNRE newer than SNLE newer than SNPE.
    method=method,
    num_workers=-1,
    num_simulations=num_sim)


n_observations = 5
observation = torch.tensor([3., -1.5])[None] + 0.5*torch.randn(n_observations, 2)

import seaborn as sns
from matplotlib import pyplot as plt

sns.scatterplot(x=observation[:, 0], y=observation[:, 1])
plt.xlabel(r'$$x_1$$')
plt.ylabel(r'$$x_2$$')
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()