using Autoencoders
import Autoencoders: AutoUNet
using Test
using Random

Random.seed!(1234)
x3 = rand(Float32, (256,256,3,1))


@testset "Autoencoders.jl" begin
    include("autoencoder-tests.jl")
end
