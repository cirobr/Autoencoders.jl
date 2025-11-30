module Autoencoders


import TinyMachines: ConvK3, ConvK1, ConvTranspK2, CBlock, MCBlock
import Flux
import Flux:
    Chain, SkipConnection, Parallel, Conv, ConvTranspose, DepthwiseConv,
    MaxPool, MeanPool, Upsample, 
    BatchNorm, Dropout,
    identity, relu, leakyrelu, relu6, σ, sigmoid, softmax,
    SamePad, kaiming_normal, rand32,
    @layer

# models
const defaultChannels = [64, 128, 256, 512, 1024]

include("./unet.jl")


end # module
