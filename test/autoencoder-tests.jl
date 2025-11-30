@info "autoencoder tests..."

modelcpu = AutoUNet()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 3, 1)
