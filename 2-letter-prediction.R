Sys.setenv("CUDA_VISIBLE_DEVICES"="0")
library(keras)

# INPUT/OUTPUT CREATION #####

# Elman created a 1,000 element sequence of the letters
# b, d, and g. Then, each letter was replaced:
# b -> ba
# d -> dii
# g -> guuu

# These six letters were represented using a six dimension
# encoding scheme.

# LETTER | CONSONANT | VOWEL | INTERRUPTED | HIGH | BACK | VOICED
#   b    |     1     |   0   |      1      |   0  |   0  |    1
#   d    |     1     |   0   |      1      |   1  |   0  |    1
#   g    |     1     |   0   |      1      |   0  |   1  |    1
#   a    |     0     |   1   |      0      |   0  |   1  |    1
#   i    |     0     |   1   |      0      |   1  |   0  |    1
#   u    |     0     |   1   |      0      |   1  |   1  |    1

# (Note that there's really only four dimensions of variation. CONSONANT
# and INTERRUPTED are redundant, and VOICED has no variation. The network
# would probably train better by removing two dimensions, but we'll follow
# what Elman did.)

generate.input <- function(length){
  bdg.letters <- sample(c('b','d','g'), length, replace=T)
  expanded.letters <- character()
  for(i in bdg.letters){
    if(i == 'b') { expanded.letters <- c(expanded.letters, c('b', 'a')) }
    if(i == 'd') { expanded.letters <- c(expanded.letters, c('d', 'i', 'i')) }
    if(i == 'g') { expanded.letters <- c(expanded.letters, c('g', 'u', 'u', 'u')) }
  }
  encoded <- sapply(expanded.letters, function(x){
    if(x=='b'){ return(c(1,0,1,0,0,1))}
    if(x=='d'){ return(c(1,0,1,1,0,1))}
    if(x=='g'){ return(c(1,0,1,0,1,1))}
    if(x=='a'){ return(c(0,1,0,0,1,1))}
    if(x=='i'){ return(c(0,1,0,1,0,1))}
    if(x=='u'){ return(c(0,1,0,1,1,1))}
  }, simplify='array')
  encoded <- t(encoded)
  return(encoded)
}

generate.output <- function(input){
  output <- matrix(0,nrow=nrow(input), ncol=6)
  output[1:(nrow(input)-1),] <- input[2:nrow(input), ]
  output[nrow(input), ] <- c(1,0,1,0,0,1) # puts a 'b' at the end.
  return(output)
}

input <- generate.input(100)
output <- generate.output(input)

input <- array_reshape(input[1:200,], dim = c(1,200,6))
output <- array_reshape(output[1:200,], dim= c(1,200,6))

### MODEL IN KERAS

model <- keras_model_sequential()
model %>%
  layer_simple_rnn(units=20, return_sequences = TRUE, input_shape=c(200,6)) %>%
  layer_dense(unit=6, activation="sigmoid")
summary(model)

model %>% compile(
  optimizer = optimizer_nadam(),
  loss = 'mean_squared_error'
)

model %>% fit(input, output, epochs=200)

result <- model %>% predict(input)

sq.error <- (result - output)^2
mse <- apply(sq.error, 2, mean)
rmse <- sqrt(mse)

decode <- function(a){
  out <- character(nrow(a))
  for(i in 1:length(out)){
    code <- a[i,]
    if(code[1] == 1 && code[4] == 0 && code[5] == 0) { out[i] <- 'b' }
    if(code[1] == 1 && code[4] == 1 && code[5] == 0) { out[i] <- 'd' }
    if(code[1] == 1 && code[4] == 0 && code[5] == 1) { out[i] <- 'g' }
    if(code[1] == 0 && code[4] == 0 && code[5] == 1) { out[i] <- 'a' }
    if(code[1] == 0 && code[4] == 1 && code[5] == 0) { out[i] <- 'i' }
    if(code[1] == 0 && code[4] == 1 && code[5] == 1) { out[i] <- 'u' }
  }
  return(out)
}

plot(rmse[1:20], type="l")
text(x=1:20, y=rmse[1:20], labels=decode(output[1,1:20,]) )

