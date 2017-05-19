
gan<-function(real_data,g_nn,d_nn,batchsize=100,epoch=100,disc_step=1,print_loss=T,display_generation_image=T,
              d_real_display=T,d_fake_print=T,display_generation_distribution=F){
  rotate <- function(x) t(apply(x, 2, rev))
  gan_model<<-list()

  x<-real_data
  numdata<-nrow(x)
  num_f<-numdata* g_nn$input_dim
  num_d<-numdata* d_nn$input_dim
  final_loss<-NULL
  for(t in 1:epoch){


    u<-1
    fak<-rnorm((num_f)) ### random noise
    fake<-matrix(fak,ncol=g_nn$input_dim)
    m <- nrow(fake)
    numbatches <- m/batchsize
    numbatches
    randperm <- sample(1:m, m)
    for(u in 1:numbatches){
      for(kk in 1:disc_step){
        ############ training discriminator

        batch_x<-fake[randperm[((u - 1) * batchsize +
                                  1):(u * batchsize)],]
        head(batch_x)
        g_nn <- nn.ff2(g_nn, batch_x,1,"g_nn") ## fake data generation from random noise
        genration<-g_nn$post[[length(g_nn$size)]]

        batch_x_2<-x[randperm[((u - 1) * batchsize +
                                 1):(u * batchsize)],]

        d_nn<-nn.ff2(d_nn,batch_x_2,1,"d_nn")
        d_real<-  d_nn$post[[length(d_nn$size)]]   ### probability real data as real
        d_fake<-nn.ff2(d_nn,genration,0,"d_nn")$post[[length(d_nn$size)]]  ### probability fake data as real

        G <- genration
        nn<-d_nn

        ######### back propagation Discriminator


        f_dn<-nn.ff2(d_nn,G,0,"d_nn") ## D(G(z))
        tr_de<-part_bp1(d_nn)
        f_de<-part_bp1(f_dn)



        for (i in 1:((length(d_nn$size)-1))) {

          dw1 <- t(tr_de[[i+1]]) %*% d_nn$post[[i]]/nrow(tr_de[[i+1 ]])
          db1 <- colMeans(tr_de[[i + 1]])
          db1 <- db1 * d_nn$learningrate

          dw2 <- t(f_de[[i +1]]) %*% f_dn$post[[i]]/nrow(f_de[[i+1]])
          db2 <- colMeans(f_de[[i + 1]])
          db2 <- db2 * d_nn$learningrate

          dw<- dw1- dw2
          dw <-  dw * d_nn$learningrate
          db <- db1-db2

          d_nn$W[[i]] <- d_nn$W[[i]] + dw #### gradien ascent
          d_nn$B[[i]] <- d_nn$B[[i]] + db
        }

        ## If you train only discriminator several times, below two probability will be 1 and 0

        if(d_real_display) {
          cat("\n Discriminator Training / Pr(real)")
          print(head(nn.ff2(d_nn,x,1,"d_nn")$post[[length(d_nn$size)]])) ## Pr(D(real))
        }
        if(d_real_display){
          cat("\n Discriminator Training / Pr(fake)")
          print(head(nn.ff2(d_nn,genration,1,"d_nn")$post[[length(d_nn$size)]]))  ## Pr(D(fake))
        }

      }
      ############ training generator
      # }

      fak<-rnorm((num_f))
      fake<-matrix(fak,ncol=g_nn$input_dim)
      # for(u in 1:numbatches){

      batch_x<-fake[randperm[((u - 1) * batchsize +
                                1):(u * batchsize)],]

      g_nn<-nn.ff2(g_nn,batch_x,0,"g_nn") ##  fake data generation from random noise
      G<-g_nn$post[[length(g_nn$size)]]
      d_nn<-nn.ff2(d_nn,G,1,"d_nn")





      ######### back propagation generator


      n <-length(c(d_nn$size,g_nn$size))-1
      d <- list()
      nn<-d_nn


      if (nn$output == "sigm") {
        d[[n]] <- nn$e * (nn$post[[length(nn$post)]] * (1 - nn$post[[length(nn$post)]]))
      }  else if (nn$output == "linear" || nn$output == "softmax") {
        d[[n]] <- nn$e
      }




      k<-0
      dl<-length(d_nn$size)-1
      dg<-length(g_nn$size)-1
      l<-dl+dg
      bp_range<-c(dl:1,dg:2)


      for (i in bp_range) {

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

        d[[l]] <- (d[[l + 1]] %*% nn$W[[i]]) * d_act

        k<-k+1
        l<-l-1
        if(k == dl){
          nn<-g_nn
        }
      }

      ## check dimension of delta
      lapply(d,dim)


      for (i in 1:dg) {
        nn<-g_nn
        dw <- t(d[[i+1]]) %*% nn$post[[i]]/nrow(t(d[[i+1]]))
        db <- colMeans(d[[i + 1]])
        db <- db * nn$learningrate
        nn$W[[i]] <- nn$W[[i]] + dw * nn$learningrate
        nn$B[[i]] <- nn$B[[i]] + db
        g_nn<-nn
      }

      if(d_real_display){
        cat("\n Generator Training / Pr(fake)")
        print(head(nn.ff2(d_nn,nn.ff2(g_nn,batch_x,0,"g_nn")$post[[length(g_nn$size)]],1,"d_nn")$post[[length(d_nn$size)]]))  ## Pr(D(fake))
      }

      if(display_generation_distribution==T){

        hist(nn.ff2(g_nn,batch_x,0,"g_nn")$post[[3]])
      }


      cat(paste0("\n",t,"-epoch ",u,"batch"))

    }

    if(display_generation_image){


      fak<-rnorm((num_f))
      fake<-matrix(fak,ncol=g_nn$input_dim)

      gg<-nn.ff2(g_nn,fake[1:100,],1,"g_nn")$post[[length(g_nn$size)]]

      # png(filename=paste0("iteration-",t,".png"))
      par(mfrow=c(3,3))
      lapply(1:9,
             function(q) image(
               rotate(matrix(unlist(gg[q,]),nrow = sqrt(d_nn$input_dim), byrow = TRUE)),
               col=grey.colors(255)        
             )
      )
      # dev.off()


    }

    if(print_loss){

      g_nn <- nn.ff2(g_nn, fake,1,"g_nn")
      genration<-g_nn$post[[length(g_nn$size)]]


      d_nn<-nn.ff2(d_nn,x,1,"d_nn")
      d_real<-  d_nn$post[[length(d_nn$size)]]

      d_fake<-nn.ff2(d_nn,genration,0,"d_nn")$post[[length(d_nn$size)]]
      final_loss<-rbind(final_loss,cbind(mean(d_real),mean(d_fake)))
      ### D_real, D_fake calculation
      ### D_real : probability real data as real
      ### D_fake : probability real data as fake
      ### In gan model, training convergence condition is D_real = D_fake = 0.5
    }
    gan_model$g_nn<<-g_nn
    gan_model$d_nn<<-d_nn

    colnames(final_loss)<-c("d_real","d_fake")

    gan_model$loss<<-final_loss
    gan_model
  }
  gan_model

}

