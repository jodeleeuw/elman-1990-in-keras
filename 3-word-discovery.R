Sys.setenv("CUDA_VISIBLE_DEVICES"="0")
library(keras)
library(stringr)

# INPUT/OUTPUT CREATION #####

# Elman created a sequence made up of
# 200 different sentences. Each sentence
# varied in length from 4-9 words. The words
# were from a lexicon of 15 possible words. 
# Each letter was encoded as a 5-dimensional
# random vector.

# Details of the sentence generator aren't 
# provided in the paper. The example sentences
# in the results (Many years ago a boy and girl
# lived by the sea. They played happily.)
# suggest that the generator produced grammatical
# sentences. But the details are murky.

# Here we'll just randomly concatenate words together
# and then we can test it using the grammatical sentence.

words <- c(
  "many",
  "years",
  "ago",
  "a",
  "boy",
  "and",
  "girl",
  "lived",
  "by",
  "the",
  "sea",
  "they",
  "played",
  "happily",
  "ocean" # this is the 15th word, unclear what it was in the original
)

# create encoding vectors randomly (may not match
# Elman's original encodings).
letter.vector <- unique(str_split(paste0(words,collapse=""), pattern="", simplify=T)[1,])

# create all 32 possible 5-element vectors
code.array <- array(data=0, dim=c(32, 5))
for(i in 0:31){
  code.array[i,] <- as.integer(intToBits(i)[1:5])
}

# sample as many of these as needed
code.array <- code.array[sample(1:32,length(letter.vector)),]

letter.encoder <- function(letter){
  letter.index <- which(letter.vector==letter)
  return(code.array[letter.index,])
}

letter.encoder('a')

# create sequences. we'll make 100 different
# inputs, each with 54 letters. This will get
# us about the same amount of training data
# as Elman (4,963 letters). We use 54 letters to 
# match the length of Elman's test string in 
# figure 6

generate.input <- function(length){
  sentence <- sample(words, length, replace=T)
  sentence.letters <- str_split(paste0(sentence, collapse=""), pattern="", simplify=T)[1,]
  sentence.letters <- sentence.letters[1:length]
  input <- array(data=0, dim=c(length, 5))
  for(i in 1:length){
    input[i,] <- letter.encoder(sentence.letters[i])
  }
  return(input)
}

generate.output <- function(input){
  output <- array(data=0, dim=c(nrow(input), 5))
  output[1:(nrow(input)-1),] <- input[2:nrow(input),]
  output[nrow(input),] <- letter.encoder('a') # add a fixed prediction at the end.
  return(output)
}

input <- array(data=0, dim=c(200, 54, 5))
output <- array(data=0, dim=c(200, 54, 5))

for(i in 1:200){
  input[i,,] <- generate.input(54)
  output[i,,] <- generate.output(input[i,,])
}

### MODEL

model <- keras_model_sequential()
model %>%
  layer_simple_rnn(units=20, input_shape=c(54,5),return_sequences = TRUE) %>%
  layer_dense(units=5, activation="sigmoid")
summary(model)

# compile the model
model %>% compile(
  optimizer = optimizer_nadam(),
  loss = 'mean_squared_error'
)

# fit the model for 100 epcohs
model %>% fit(input, output, epochs=100, validation_split = 0.2)


### PREDICTIONS

# Use the example sentence from Elman.
# Need an extra letter at the start so that prediction starts with the "m" in many.
test.letters <- 'amanyyearsagoaboyandgirllivedbytheseatheyplayedhappily'
test.letters <- str_split(test.letters, pattern="", simplify=T)[1,]
input.novel <- array(data=0, dim=c(54,5))
for(i in 1:54){
  input.novel[i,] <- letter.encoder(test.letters[i])
}
output.novel <- generate.output(input.novel)


input.novel <- array_reshape(input.novel, dim = c(1,54,5))
output.novel <- array_reshape(output.novel, dim= c(1,54,5))

prediction <- model %>% predict(input.novel)

# calculate RMSE, averaging over the 5 outputs per timestep
sq.error <- (prediction - output.novel)^2
mse <- apply(sq.error, 2, mean)
rmse <- sqrt(mse)

# recreate the figure from the paper
plot(1:54, rmse, type="l")
text(x=1:54, y=rmse, labels=test.letters[2:54])
