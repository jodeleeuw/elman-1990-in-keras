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
# inputs, each with 50 letters. This will get
# us about the same amount of training data
# as Elman (4,963 letters).

generate.input <- function(length){
  sentence <- sample(words, length, replace=T)
}
