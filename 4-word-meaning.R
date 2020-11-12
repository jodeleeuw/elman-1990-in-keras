

word.categories <- list(
  noun.human = c('man', 'woman', 'boy', 'girl'),
  noun.animal = c('cat', 'dog', 'mouse', 'lion', 'dragon', 'monster'),
  noun.inanimate = c('book', 'rock', 'glass', 'plate', 'sandwich', 'cookie', 'bread', 'car'),
  noun.agressive = c('dragon', 'monster', 'lion'),
  noun.fragile = c('glass', 'plate'),
  noun.food = c('sandwich', 'cookie', 'bread'),
  verb.intransitive = c('think', 'sleep', 'exist'),
  verb.transitive = c('smell', 'move', 'see', 'break', 'smash', 'like', 'chase', 'eat'),
  verb.agentpatient = c('move', 'break', 'smell', 'see'),
  verb.perception = c('see', 'smell'),
  verb.destroy = c('break', 'smash'),
  verb.eat = c('eat')
)

word.index <- unique(unlist(word.categories, use.names = F))

sentence.templates <- array(data=NA, dim=c(15,3)) 
sentence.templates[1,] = c("noun.human", "verb.eat", "noun.food")
sentence.templates[2,] = c("noun.human", "verb.perception", "noun.inanimate")
sentence.templates[3,] = c("noun.human", "verb.destroy", "noun.fragile")
sentence.templates[4,] = c("noun.human", "verb.intransitive", NA)
sentence.templates[5,] = c("noun.human", "verb.agentpatient", "noun.inanimate")
sentence.templates[6,] = c("noun.human", "verb.agentpatient", NA)
sentence.templates[7,] = c("noun.animal", "verb.eat", "noun.food")
sentence.templates[8,] = c("noun.animal", "verb.transitive", "noun.animal")
sentence.templates[9,] = c("noun.animal", "verb.agentpatient", "noun.inanimate")
sentence.templates[10,] = c("noun.animal", "verb.agentpatient", NA)
sentence.templates[11,] = c("noun.inanimate", "verb.agentpatient", NA)
sentence.templates[12,] = c("noun.agressive", "verb.destroy", "noun.fragile")
sentence.templates[13,] = c("noun.agressive", "verb.eat", "noun.human")
sentence.templates[14,] = c("noun.agressive", "verb.eat", "noun.animal")
sentence.templates[15,] = c("noun.agressive", "verb.eat", "noun.food")

# create input and output
generate.input <- function(length){
  # pick random sentence frames
  sentences <- sentence.templates[sample(1:15, ceiling(length/2), replace=T),]
  
  # flatten sentences to single continuous stream
  flat.sentences <- as.vector(t(sentences))
  flat.sentences <- flat.sentences[!is.na(flat.sentences)]
  
  # words
  words <- sapply(flat.sentences, function(word.cat){
    return(sample(word.categories[[word.cat]], 1))
  }, USE.NAMES = FALSE)
  
  # one-hot encoding
  word.int <- as.numeric(factor(words, levels = word.index))
  word.vectors <- to_categorical(word.int, 31)
  
  # truncate to target length
  word.vectors <- word.vectors[1:length,]
  return(word.vectors)
}

generate.output <- function(input){
  output <- array(data=0, dim=c(nrow(input), ncol(input)))
  output[1:(nrow(input)-1),] <- input[2:nrow(input),]
  output[nrow(input),] <- input[1,] # add a fixed prediction at the end.
  output <- apply(output, 1, function(x){ return (which(x==1)) })
  return(output)
}

train.sequences <- 100
sequence.length <- 280

input <- array(data=0, dim=c(train.sequences, sequence.length, 31))
output <- array(data=0, dim=c(train.sequences, sequence.length, 1))

for(i in 1:train.sequences){
  input[i,,] <- generate.input(sequence.length)
  output[i,,] <- generate.output(input[i,,])
}

#### MODEL ####

model <- keras_model_sequential()
model %>%
  layer_simple_rnn(name="hidden",units=150, return_sequences = TRUE, input_shape=c(sequence.length, 31)) %>%
  layer_dense(units=31, activation='softmax')
summary(model)

model %>%
  compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = optimizer_nadam(),
    metrics=c('accuracy')
  )

model %>% fit(input, output, epochs=20, validation_split=0.2)

#### PREDICTIONS ####

# once the model is trained, run the input again and keep track of hidden layer
# representations for each word.

hidden.layer.model <- keras_model(inputs=model$input, outputs=model$get_layer('hidden')$output)

hidden.reps <- hidden.layer.model %>% predict(input)

# average all of the reps for each unique input

flat.hidden.reps <- array_reshape(hidden.reps, dim=c(28000, 150))
flat.input <- array_reshape(input, dim=c(28000, 31))
input.word <- apply(flat.input, 1, function(x){ return (which(x==1)) }) - 1

unique(input.word)

m <- t(sapply(1:29, function(x){ return(apply(flat.hidden.reps[input.word==x,],2,mean))}))

rownames(m) <- word.index
# clustering representation

dissimilarity <- dist(m)

hc1 <- hclust(dissimilarity, method = "complete" )

plot(hc1, xlab="", main=NULL, sub="", ylab=NULL, yaxt="n")
