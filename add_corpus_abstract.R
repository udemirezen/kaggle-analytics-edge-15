# Prepare corpus using snippet
corpusAbstract <- Corpus(VectorSource(c(newsTrain$Abstract, newsTest$Abstract)))
corpusAbstract <- tm_map(corpusAbstract, tolower)
corpusAbstract <- tm_map(corpusAbstract, PlainTextDocument)
corpusAbstract <- tm_map(corpusAbstract, removePunctuation)
corpusAbstract <- tm_map(corpusAbstract, removeWords, stopwords("english"))
corpusAbstract <- tm_map(corpusAbstract, stripWhitespace)
corpusAbstract <- tm_map(corpusAbstract, stemDocument)

# Generate term matrix
dtmAbstract <- DocumentTermMatrix(corpusAbstract)
sparseAbstract <- removeSparseTerms(dtmAbstract, 0.99)
abstractWords <- as.data.frame(as.matrix(sparseAbstract))

colnames(abstractWords) <- make.names(colnames(abstractWords))
colnames(abstractWords) <- paste0("A_", colnames(abstractWords))

abstractWordsTrain <- head(abstractWords, nrow(newsTrain))
abstractWordsTest <- tail(abstractWords, nrow(newsTest))

# Add to dataframes
newsTrain <- cbind(newsTrain, abstractWordsTrain)
newsTest <- cbind(newsTest, abstractWordsTest)

# Explore a bit
# ...

# Remove original text variables
newsTrain$Abstract <- NULL
newsTest$Abstract <- NULL
