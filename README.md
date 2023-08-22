# TrashGrad
A grabage numpy/cupy-based autograd engine inspired by micrograd built so I can get a better understanding of autograd and gradient flow by buidling it. It is pretty slow, in the worst case up to 10x slower than PyTorch (because I had to write some python for loops in the Convolution backward)
