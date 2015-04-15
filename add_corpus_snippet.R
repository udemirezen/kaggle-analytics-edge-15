# Prepare corpus using snippet
corpusSnippet <- Corpus(VectorSource(c(newsTrain$Snippet, newsTest$Snippet)))
corpusSnippet <- tm_map(corpusSnippet, tolower)
corpusSnippet <- tm_map(corpusSnippet, PlainTextDocument)
corpusSnippet <- tm_map(corpusSnippet, removePunctuation)
corpusSnippet <- tm_map(corpusSnippet, removeWords, stopwords("english"))
corpusSnippet <- tm_map(corpusSnippet, stripWhitespace)
corpusSnippet <- tm_map(corpusSnippet, stemDocument)

# Generate term matrix
dtmSnippet <- DocumentTermMatrix(corpusSnippet)
sparseSnippet <- removeSparseTerms(dtmSnippet, 0.999)
snippetWords <- as.data.frame(as.matrix(sparseSnippet))

colnames(snippetWords) <- make.names(colnames(snippetWords))
colnames(snippetWords) <- paste0("S_", colnames(snippetWords))

snippetWordsTrain <- head(snippetWords, nrow(newsTrain))
snippetWordsTest <- tail(snippetWords, nrow(newsTest))

# Add to dataframes
newsTrain <- cbind(newsTrain, snippetWordsTrain)
newsTest <- cbind(newsTest, snippetWordsTest)

# Explore a bit
# ...

# Remove original text variables
newsTrain$Snippet <- NULL
newsTest$Snippet <- NULL
