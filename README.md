![alt text](https://github.com/cirobr/TinyMachines.jl/blob/main/images/logo-name-tm.png?raw=true)

# Autoencoders

[![Build Status](https://github.com/cirobr/Autoencoders.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/Autoencoders.jl/actions/workflows/CI.yml?query=branch%3Amain)

 Auto-encoders for TinyMachines.jl computer vision models.

 ## Syntax

With no arguments, all models accept 3-channels Float32 input and deliver 3-channels output with sigmoid output activation.

```
model = AutoUNet()
model = AutoUNet(3,3)
```

## Models

```
AutoUNet(3, 3;             # input/output channels
    activation = relu,     # activation function
)
```

## Constructors

Constructors are underlying models that allow access to a multitude of hyperparameters. Each model from above has been build with the aid of these constructors, where hyperparameters are chosen for performance.

```
autounet(3, 1;                            # input/output channels
    activation = relu,                    # activation function
    alpha = 1,                            # channels divider
    edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
    ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)
```
