

word.categories <- list(
  noun.human = c('man', 'woman', 'boy', 'girl'),
  noun.animal = c('cat', 'mouse', 'lion', 'dragon', 'monster'),
  noun.inanimate = c('book', 'rock', 'glass', 'plate', 'sandwich', 'cookie', 'bread', 'car'),
  noun.agressive = c('dragon', 'monster', 'lion'),
  noun.fragile = c('glass', 'plate'),
  noun.food = c('sandwich', 'cookie', 'bread'),
  verb.intransitive = c('think', 'sleep', 'exist'),
  verb.transitive = c('smell', 'move', 'see', 'break', 'smash', 'like', 'chase', 'eat'),
  verb.agpat = c('move', 'break', 'chase'),
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
sentence.templates[5,] = c("noun.human", "verb.agpat", "noun.inanimate")
sentence.templates[6,] = c("noun.human", "verb.agpat", NA)
sentence.templates[7,] = c("noun.animal", "verb.eat", "noun.food")
sentence.templates[8,] = c("noun.animal", "verb.transitive", "noun.animal")
sentence.templates[9,] = c("noun.animal", "verb.agpat", "noun.inanimate")
sentence.templates[10,] = c("noun.animal", "verb.agpat", NA)
sentence.templates[11,] = c("noun.inanimate", "verb.agpat", NA)
sentence.templates[12,] = c("noun.agressive", "verb.destroy", "noun.fragile")
sentence.templates[13,] = c("noun.agressive", "verb.eat", "noun.human")
sentence.templates[14,] = c("noun.agressive", "verb.eat", "noun.animal")
sentence.templates[15,] = c("noun.agressive", "verb.eat", "noun.food")

# create 10,000 random sentence frames
sentences <- sentence.templates[sample(1:15, 10000, replace=T),]

# flatten sentences to single continuous stream
flat.sentences <- as.vector(t(sentences))
flat.sentences <- flat.sentences[!is.na(flat.sentences)]

# words
words <- sapply(flat.sentences, function(word.cat){
  return(sample(word.categories[[word.cat]], 1))
}, USE.NAMES = FALSE)

# one-hot encoding

word.int <- as.numeric(factor(words, levels = word.index))

a <- keras::k_one_hot(array_reshape(word.int, dim=c(1,length(word.int))), 31)
