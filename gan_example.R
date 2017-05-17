setwd("C:\\gan\\R")
source("gan.R")
source("generator.R")
source("nn.ff2.R")
source("nn.model.R")
source("part_bp1.R")

source("relu.R")
source("sigm.R")

### load mnist data sample(2000)
load("sample_mnist.RData")
x<-train
x<-x/255
x<-as.matrix(x)
### initialize model
g_nn<-nn_model(input_dim=64,hidden=85,output_dim=784,learningrate=0.1,
               activationfun="relu",output="sigm" )
d_nn<-nn_model(input_dim=784,hidden=85,output_dim=1,learningrate=0.1,
               activationfun="relu",output="sigm" )

numdata<-2000
num_f<-numdata* g_nn$input_dim
num_d<-numdata* d_nn$input_dim

### traning GANs
ganmodel<-gan(x,g_nn,d_nn,batchsize = 200,epoch = 1000,disc_step=1,display_generation_distribution = F,display_generation_image = T)
### If you stop training, stopped model will be saved "gan_model".
gan_model$loss


genration<-generator(ganmodel,100)
## or if you stop traning
genration<-generator(gan_model,100)

hist(generation)

rotate <- function(x) t(apply(x, 2, rev))

par(mfrow=c(3,3))
lapply(1:9,
       function(q) image(
         rotate(matrix(unlist(generator[q,]),nrow = 28, byrow = TRUE)),
         col=grey.colors(255)
       )
)
