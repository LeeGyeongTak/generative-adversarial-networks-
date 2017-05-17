part_bp1<- function(nn){
  n <- length(nn$size)
  d <- list()
  if (nn$output == "sigm") {


    d[[n]] <- nn$e * (nn$post[[n]] * (1 - nn$post[[n]]))
  }  else if (nn$output == "linear" || nn$output == "softmax") {
    d[[n]] <- nn$e
  }

  for (i in (n - 1):2) {
    if (nn$activationfun == "linear") {
      d_act <- nn$post[[i]]
    }
    if (nn$activationfun == "sigm") {
      d_act <- nn$post[[i]] * (1 - nn$post[[i]])
    }
    if (nn$activationfun == "relu") {
      d_act <-  ifelse(nn$post[[i]]>0,1,0)
    }
    else if (nn$activationfun == "tanh") {
      d_act <- 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn$post[[i]]^2)
    }


    d[[i]] <- (d[[i + 1]] %*% nn$W[[i]]) * d_act


  }
  d
}
