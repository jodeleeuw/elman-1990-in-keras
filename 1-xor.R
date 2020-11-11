Sys.setenv("CUDA_VISIBLE_DEVICES"="0")
library(keras)

# INPUT/OUTPUT CREATION #####

# Elman presented a 3,000 item sequence 600 times.
# We'll take advantage of batch parallelization and
# run 100 different 300-item sequences 60 times.

sequence.length <- 300 # must be divisible by 3
num.sequences <- 100 # unique sequences to train on

generate.input <- function(pairs){
  output <- numeric(pairs*3)
  for(i in 0:(pairs-1)){
    output[1+(i*3)] <- sample(c(0,1),1)
    output[2+(i*3)] <- sample(c(0,1),1)
    output[3+(i*3)] <- xor(output[1+(i*3)], output[2+(i*3)])
  }
  return(output)
}

generate.output <- function(input){
  return(c(input[2:length(input)], sample(c(0,1), 1)))
}

input <- array(data=0, dim=c(num.sequences, sequence.length, 1))
output <- array(data=0, dim=c(num.sequences, sequence.length, 1))

for(i in 1:num.sequences){
  input[i,,] <- generate.input(sequence.length/3)
  output[i,,] <- generate.output(input[i,,])
}

# SIMPLE RNN MODEL #####

# 1 input unit, 2 hidden units, 1 output unit

model <- keras_model_sequential()
model %>%
  layer_simple_rnn(units=2, return_sequences=T, input_shape=c(sequence.length, 1)) %>%
  layer_dense(units=1,  activation = "sigmoid")
summary(model)

model %>% compile(
  optimizer=optimizer_nadam(lr=0.02),
  loss='mean_squared_error',
  metrics=c('accuracy')
)

model %>% fit(input, output, epochs=60)


# GENERATE NOVEL PREDICTION ####

test.sequences <- 200

novel.input <- array(data=0, dim=c(test.sequences, sequence.length, 1))
novel.output <- array(data=0, dim=c(test.sequences, sequence.length, 1))

for(i in 1:test.sequences){
  novel.input[i,,] <- generate.input(sequence.length/3)
  novel.output[i,,] <- generate.output(novel.input[i,,])
}

prediction <- model %>% predict(novel.input)

sq.err <- (novel.output - prediction)^2
mse <- apply(sq.err,2,mean)
rmse <- sqrt(mse)

wrap.around <- matrix(data=rmse, ncol=12, byrow = T)
m <- apply(wrap.around,2,mean)

plot(m, type="o", xlab="Cycle", ylab="Error")

