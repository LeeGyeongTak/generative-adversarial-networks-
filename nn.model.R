nn_model<-function( input_dim=64,hidden=85,output_dim=784,learningrate=0.05,
                    activationfun="relu",output="sigm" ){
  size <- c(input_dim, hidden, output_dim)
  vW2 <- list()
  vB2 <- list()
  W2 <- list()
  B2 <- list()
  for (i in 2:length(size)) {
    W2[[i - 1]] <- matrix(runif(size[i] * size[i - 1],
                                min = -0.1, max = 0.1), c(size[i], size[i - 1]))
    B2[[i - 1]] <- runif(size[i], min = --0.1, max = 0.1)
    vW2[[i - 1]] <- matrix(rep(0, size[i] * size[i - 1]),
                           c(size[i], size[i - 1]))
    vB2[[i - 1]] <- rep(0, size[i])
  }


  nn <- list(input_dim = input_dim, output_dim = output_dim,
             hidden = hidden, size = size, activationfun = activationfun,
             learningrate = learningrate,
             output = output, W = W2, vW = vW2, B = B2, vB = vB2)
  nn

}
