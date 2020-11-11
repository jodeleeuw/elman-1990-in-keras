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

# (Note that there's really only three dimensions of variation. CONSONANT,
# VOWEL, and INTERRUPTED are redundant with each other, and VOICED has 
# no variation. The network would probably train better by removing three 
# dimensions, but we'll follow what Elman did.)

generate.input <- function(length){
  # first, randomly sample b d g to make a vector of letters
  bdg.letters <- sample(c('b','d','g'), length, replace=T)
  
  # second, add the vowel sounds to each letter
  expanded.letters <- character()
  for(i in bdg.letters){
    if(i == 'b') { expanded.letters <- c(expanded.letters, c('b', 'a')) }
    if(i == 'd') { expanded.letters <- c(expanded.letters, c('d', 'i', 'i')) }
    if(i == 'g') { expanded.letters <- c(expanded.letters, c('g', 'u', 'u', 'u')) }
  }
  
  # finally, replace the letters with their 6-dimensional code
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
  # create a matrix to hold the right size output
  output <- matrix(0,nrow=nrow(input), ncol=6)
  # copy the input, but move it up one step
  output[1:(nrow(input)-1),] <- input[2:nrow(input), ]
  # add something to the final spot
  output[nrow(input), ] <- c(1,0,1,0,0,1) # puts a 'b' at the end.
  return(output)
}

# Elman used a 1,000 letter (b, d, g) sequence, shown to the network
# 200 times. Since the average consonant has 2 vowels, we can estimate
# that the total sequence length was 3,000. We'll use 10 unique 300-length
# sequences instead, to speed up training a bit.

input <- array(0, dim=c(10, 300, 6))
output <- array(0, dim=c(10, 300, 6))

for(i in 1:10){
  inp <- generate.input(150) # 150 guarantees a minimum expanded length of 300.
  out <- generate.output(inp)
  input[i,,] <- inp[1:300,] # truncate the sequence to first 300 elements
  output[i,,] <- out[1:300,]
}

### MODEL IN KERAS

# create the model (6 inputs, 20 recurrent hidden units, 6 outputs)
model <- keras_model_sequential()
model %>%
  layer_simple_rnn(units=20, return_sequences = TRUE, input_shape=c(300,6)) %>%
  layer_dense(unit=6, activation="sigmoid")
summary(model)

# compile the model
model %>% compile(
  optimizer = optimizer_nadam(),
  loss = 'mean_squared_error'
)

# fit the model for 200 epcohs
model %>% fit(input, output, epochs=200, validation_split = 0.2)

### TESTING THE MODEL #####

# generate novel input sequence
input.novel <- generate.input(150)
output.novel <- generate.output(input.novel)

# shape the novel input/output into array of dimension
# c(1, 300, 6). First entry = number of sequences,
# second = timesteps, third = input length
input.novel <- array_reshape(input.novel[1:300,], dim = c(1,300,6))
output.novel <- array_reshape(output.novel[1:300,], dim= c(1,300,6))

# run the model on the new input
result <- model %>% predict(input.novel)

# calculate RMSE, averaging over the 6 outputs per timestep
sq.error <- (result - output.novel)^2
mse <- apply(sq.error, 2, mean)
rmse <- sqrt(mse)

# this function decodes the 6-dimensional coding back to letters
# we need it to plot the letters on the graph
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

# plot the RMSE over the first 20 timesteps
# add the text so we can visualize how error
# changes as a function of sequence predictability.
plot(rmse[1:20], type="l")
text(x=1:20, y=rmse[1:20], labels=decode(output.novel[1,1:20,]))
