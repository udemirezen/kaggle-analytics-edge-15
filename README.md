# The Analytics Edge - Kaggle Competition 2015
Jose A. Dianes  
13 April 2015  

# Task description  

# Data loading and preparation  


```r
library(tm)
```

Let's start by reading the trian and test data into the corresponding data frames.  


```r
newsTrain <- read.csv("data/NYTimesBlogTrain.csv", stringsAsFactors=FALSE)
newsTest <- read.csv("data/NYTimesBlogTest.csv", stringsAsFactors=FALSE)
summary(newsTrain)
```

```
##    NewsDesk         SectionName        SubsectionName    
##  Length:6532        Length:6532        Length:6532       
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##    Headline           Snippet            Abstract        
##  Length:6532        Length:6532        Length:6532       
##  Class :character   Class :character   Class :character  
##  Mode  :character   Mode  :character   Mode  :character  
##                                                          
##                                                          
##                                                          
##    WordCount         PubDate             Popular          UniqueID   
##  Min.   :    0.0   Length:6532        Min.   :0.0000   Min.   :   1  
##  1st Qu.:  187.0   Class :character   1st Qu.:0.0000   1st Qu.:1634  
##  Median :  374.0   Mode  :character   Median :0.0000   Median :3266  
##  Mean   :  524.4                      Mean   :0.1673   Mean   :3266  
##  3rd Qu.:  723.2                      3rd Qu.:0.0000   3rd Qu.:4899  
##  Max.   :10912.0                      Max.   :1.0000   Max.   :6532
```

From the summary we can see that we have several fields we could use to train
our models.  

# A simple bag-of-words model  

As a first approach, we will use bag-of-words models for teh headline text.  

## Preparing the corpus  

In order to build the corpus, we will go through the usual `tm` package calls.  


```r
corpusHeadline <- Corpus(VectorSource(c(newsTrain$Headline, newsTest$Headline)))
corpusHeadline <- tm_map(corpusHeadline, tolower)
corpusHeadline <- tm_map(corpusHeadline, PlainTextDocument)
corpusHeadline <- tm_map(corpusHeadline, removePunctuation)
corpusHeadline <- tm_map(corpusHeadline, removeWords, stopwords("english"))
corpusHeadline = tm_map(corpusHeadline, stemDocument)
```

Now we are ready to convert our corpus to a DocumentTermMatrix, remove sparse 
terms, and turn it into a data frame. We selected one particular threshold to 
remove sparse terms. Later on we must try different ones.  


```r
dtm <- DocumentTermMatrix(corpusHeadline)
sparse <- removeSparseTerms(dtm, 0.99)
headlineWords = as.data.frame(as.matrix(sparse))
```

Let's make sure our variable names are okay for R.  


```r
colnames(headlineWords) = make.names(colnames(headlineWords))
```

## Training the models  

### Logistic regression  

### CART  

### Random forest  


## Model evaluation  


# A richer bag of words model  

# Cluster helps us understand data  

# Either cluster then predict (multiple predictors), or use most frequent terms from each cluster (term weighting)  

# Experiment: using separate models for each section and average them manually with different weights   



