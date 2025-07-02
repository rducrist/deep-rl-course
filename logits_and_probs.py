# visualize_action_distribution(probs)
'''
Learning here: The NN output logits, which can be arbitrary number. We use softmax to map them to a [0,1] interval. 
Note that the mapping is non-linear!
'''
import torch 
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

input = torch.randn(1, 3)  # shape: (1, num_actions)
softmax = nn.Softmax(dim=1)
output = softmax(input)

# Remove batch dimension
logits = input.squeeze().detach().numpy()
probs = output.squeeze().detach().numpy()

# Bar plot setup
x = np.arange(len(logits))  # action indices
width = 0.4

plt.bar(x - width/2, logits, width, label='Logits', color='orange')
plt.bar(x + width/2, probs, width, label='Softmax Probabilities', color='skyblue')

# Labels and formatting
plt.xlabel('Action')
plt.ylabel('Value')
plt.title('Logits vs. Softmax Probabilities')
plt.xticks(x)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()