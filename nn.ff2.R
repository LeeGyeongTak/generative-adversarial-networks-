nn.ff2<-function (nn, batch_x, true_value,dd)
{
  m <- nrow(batch_x)

  nn$post[[1]] <- batch_x
  for (i in 2:(length(nn$size) - 1)) {
    nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) +
                       nn$B[[i - 1]])
    if (nn$activationfun == "sigm") {
      nn$post[[i]] <- sigm(nn$pre[[i]])
    }
    else if (nn$activationfun == "tanh") {
      nn$post[[i]] <- tanh(nn$pre[[i]])
    }
    else if (nn$activationfun == "relu") {
      nn$post[[i]] <- relu(nn$pre[[i]])
    }
    else if (nn$activationfun == "linear") {
      nn$post[[i]] <- (nn$pre[[i]])
    }
    else {
      stop("unsupport activation function!")
    }

  }
  i <- length(nn$size)
  nn$pre[[i]] <- t(nn$W[[i - 1]] %*% t(nn$post[[(i - 1)]]) +
                     nn$B[[i - 1]])
  if (nn$output == "sigm") {
    nn$post[[i]] <- sigm(nn$pre[[i]])

    if(true_value==1){
      if(dd == "d_nn"){
        nn$e<-1/(nn$post[[i]] )
      }
      if(dd == "g_nn"){
        nn$e<-1/nn.ff2(d_nn,nn$post[[i]],1,"d_nn")$post[[3]]
      }
    }else{

      nn$e<-1/(1-nn$post[[i]] )


    }

  } else{
    stop("unsupport output function!")

  }
  nn
}
