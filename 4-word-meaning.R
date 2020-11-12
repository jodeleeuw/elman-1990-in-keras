## PART 4: DISCOVERING LEXICAL CLASSES FROM WORD ORDER

# Elman created a set of 29 words and 12 lexical classes. Some words appear in multiple classes.
# Here I've created the classes in a list and assigned the vector of words to each. Note that
# the full list of class membership is not published in the paper, so I made some guesses.

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

# We're going to end up using various integer-based representations for the words,
# so this next line creates a vector containing all 29 words. We'll assign the words
# to integers based on their position in this list.
word.index <- unique(unlist(word.categories, use.names = F))

# Elman defined 15 different sentence templates, which consist of 2 or 3 classes
# of words. Here I've created an array where rows are sentence templates and columns
# are the word classes at position 1, 2, and 3 in that template. NA = no word.
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

# INPUT GENERATOR
# The input to the model is a sequence of words, created by picking sentence 
# templates and randomly sampling the words of each class to form a sentence. 
# Sentences are arranged continuously (no gap). Words are represented by a 
# 31-bit one hot encoding. There are only 29 words, so 2 bits are always 0.
# Elman used these bits for other simulations after the model was trained.
# To conform with consistent batch sizes in the keras model, we'll create input
# based on a length parameter, where length is the number of words in the total 
# sequence.

generate.input <- function(length){
  # Pick random sentence templates. Get enough to guarantee that even if all
  # 2-words sentences are true, we still have the minimum length.
  sentences <- sentence.templates[sample(1:15, ceiling(length/2), replace=T),]
  
  # flatten sentences to single continuous stream
  flat.sentences <- as.vector(t(sentences))
  flat.sentences <- flat.sentences[!is.na(flat.sentences)] # removes the NA words
  
  # Map the lexical classes to words by randomly picking a word from the class
  # for each entry in the flat.sentences vector.
  words <- sapply(flat.sentences, function(word.cat){
    return(sample(word.categories[[word.cat]], 1))
  }, USE.NAMES = FALSE)
  
  # Encode the input using one hot vectors.
  word.int <- as.numeric(factor(words, levels = word.index))
  word.vectors <- to_categorical(word.int, 31)
  
  # Truncate the entire sequence to target length
  word.vectors <- word.vectors[1:length,]
  return(word.vectors)
}

generate.output <- function(input){
  # Create an empty array that matches the dimensions of the input
  output <- array(data=0, dim=c(nrow(input), ncol(input)))
  # Copy the input to the output, moving all items up one spot 
  # to match the prediction task of the network.
  output[1:(nrow(input)-1),] <- input[2:nrow(input),]
  output[nrow(input),] <- input[1,] # add a fixed prediction at the end.
  # our model will use a softmax layer as the final output, so keras
  # allows us to pass the target output as integers rather than one
  # hot vectors. This converts the one hot vectors back to integers.
  output <- apply(output, 1, function(x){ return (which(x==1)) })
  return(output)
}

# Elman used a single input stream of 27,534 words. We'll split this up
# into 100 streams of 280 words to take advantage of batch processing.

train.sequences <- 100
sequence.length <- 280

input <- array(data=0, dim=c(train.sequences, sequence.length, 31))
output <- array(data=0, dim=c(train.sequences, sequence.length, 1))

for(i in 1:train.sequences){
  input[i,,] <- generate.input(sequence.length)
  output[i,,] <- generate.output(input[i,,])
}

#### MODEL ####
# Elman's model had 31 input nodes, 150 hidden recurrent nodes, and 31 output nodes.
# The output layer in Elman's model did not use the sum-to-1 softmax constraint, but
# he probably would have if it had been an option. He identifies some problems with
# the fact that activations don't sum to 1 in his analysis, and ends up analyzing the
# data in a way that is more consistent with using a softmax activation. As you'll see
# we can replicate the results of the paper really well and save ourselves some analysis
# headaches by just using the softmax function here.

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

# The critical analysis in the paper is not the predictive success of the model
# at the output layer, but rather the similarity of the hidden layer activations
# across different classes of words. Now that we've trained the model, we can create
# a new keras model that uses just the input -> hidden layer of our trained model
# and returns the entire sequence of recurrent node activations.

hidden.layer.model <- keras_model(inputs=model$input, outputs=model$get_layer('hidden')$output)

# Run the input through the model again, but now the output will be the hidden layer
# activation.

hidden.reps <- hidden.layer.model %>% predict(input)

# The next goal is to find the average activation for each of the different words.
# Step one is arranging the hidden node activations and the input into an array
# where each row is a word and each column is the activation.

flat.hidden.reps <- array_reshape(hidden.reps, dim=c(train.sequences*sequence.length, 150))
flat.input <- array_reshape(input, dim=c(train.sequences*sequence.length, 31))

# Now we decode the input back into the integer-based representation of the word.
input.word <- apply(flat.input, 1, function(x){ return (which(x==1)) }) - 1

# Finally, we loop through all 29 word indices, finding all of the hidden layer activations
# that are from when that word was presented, and averaging over the activations for each.
# This gives us a 29 x 150 matrix, where rows are words and columns are the average activations
# for each hidden unit.
m <- t(sapply(1:29, function(x){ return(apply(flat.hidden.reps[input.word==x,],2,mean))}))

# We can label the rows with the actual word that was presented.
rownames(m) <- word.index

# R has some great tools for calculating distance matrices. We can provide the dist() function
# with a matrix where each row is an item and the columns are features, and it will return 
# an object with all the pairwise distance calculations.
dissimilarity <- dist(m)

# We can then apply a hierarchical clustering algorithm to find groupings that put items with
# similar features (low distance) together.
hc1 <- hclust(dissimilarity, method = "complete" )

# We can draw the dendrogram of this clustering to create a replica of Figure 7 in Elman 1990.
plot(hc1, xlab="", main=NULL, sub="", ylab=NULL, yaxt="n")
