
generator<-function(ganmodel,n_g_data){

  generat<-ganmodel$g_nn

  fak<-rnorm((n_g_data)*generat$input_dim)
  fake<-matrix(fak,ncol=generat$input_dim)
  nn.ff2(generat,fake,1,"g_nn")$post[[3]]

}
