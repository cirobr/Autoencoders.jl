struct autounet
    encoder::Chain
    decoder::Chain
end
@layer autounet

function autounet(ch_in::Int=3, ch_out::Int=3;        # input/output channels
               activation::Function = relu,          # activation function
               alpha::Int           = 1,             # channels divider
               edrops = (0.0, 0.0, 0.0, 0.0, 0.0),   # dropout rates
               ddrops = (0.0, 0.0, 0.0, 0.0),        # dropout rates
)

    chs = defaultChannels .÷ alpha

    # encoder
    e1 = Chain(CBlock(ch_in, chs[1], activation),   Dropout(edrops[1]))
    e2 = Chain(MCBlock(chs[1], chs[2], activation), Dropout(edrops[2]))
    e3 = Chain(MCBlock(chs[2], chs[3], activation), Dropout(edrops[3]))
    e4 = Chain(MCBlock(chs[3], chs[4], activation), Dropout(edrops[4]))
    e5 = Chain(MCBlock(chs[4], chs[5], activation), Dropout(edrops[5]))

    # decoder
    d4 = Chain(ConvTranspK2(chs[5], chs[4], activation; stride=2), ConvK3(chs[4], chs[4], activation, stride=1), Dropout(ddrops[4]))
    d3 = Chain(ConvTranspK2(chs[4], chs[3], activation; stride=2), ConvK3(chs[3], chs[3], activation, stride=1), Dropout(ddrops[3]))
    d2 = Chain(ConvTranspK2(chs[3], chs[2], activation; stride=2), ConvK3(chs[2], chs[2], activation, stride=1), Dropout(ddrops[2]))
    d1 = Chain(ConvTranspK2(chs[2], chs[1], activation; stride=2), ConvK3(chs[1], chs[1], activation, stride=1), Dropout(ddrops[1]))
    d0 = ConvK1(chs[1], ch_out)

    # output chains
    encoder = Chain(e1=e1, e2=e2, e3=e3, e4=e4, e5=e5)
    decoder = Chain(d4=d4, d3=d3, d2=d2, d1=d1, d0=d0)

    return autounet(encoder, decoder)   # struct output
end


function (m::autounet)(x::AbstractArray{Float32,4})
    enc1 = m.encoder[:e1](x)
    enc2 = m.encoder[:e2](enc1)
    enc3 = m.encoder[:e3](enc2)
    enc4 = m.encoder[:e4](enc3)
    enc5 = m.encoder[:e5](enc4)

    dec4 = m.decoder[:d4](enc5)
    dec3 = m.decoder[:d3](dec4)
    dec2 = m.decoder[:d2](dec3)
    dec1 = m.decoder[:d1](dec2)

    logits = m.decoder[:d0](dec1)

    feature_maps = [enc1, enc2, enc3, enc4, enc5,     # encoder[1:5]
                    dec4, dec3, dec2, dec1, logits]   # decoder[6:10]
    return feature_maps
end


function AutoUnet(ch_in::Int=3, ch_out::Int=3;    # input/output channels
               activation::Function = relu,    # activation function
)
    model = autounet(ch_in, ch_out;
                  activation=activation,
                  alpha=1,
                  edrops=(0.0, 0.0, 0.1, 0.2, 0.25),
                  ddrops=(0.0, 0.0, 0.1, 0.2),
    )
    act = x -> σ.(x)
    return Chain(model, x->x[end], act)
end
