@info "autoencoder tests..."

modelcpu = AutoUnet()
yhat  = modelcpu(x3)
@test size(yhat) == (256, 256, 3, 1)
