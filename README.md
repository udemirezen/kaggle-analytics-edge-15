# The Analytics Edge - Kaggle Competition 2015
Jose A. Dianes  
13 April 2015  

# Task description  

# Data loading and preparation  


```r
library(tm)
library(ROCR)
library(rpart)
library(rpart.plot)
library(caTools)
library(randomForest)
```

Let's start by reading the train and test data into the corresponding data frames.  


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
corpusHeadline <- tm_map(corpusHeadline, stemDocument)
```

Now we are ready to convert our corpus to a DocumentTermMatrix, remove sparse 
terms, and turn it into a data frame. We selected one particular threshold to 
remove sparse terms. Later on we must try different ones.  


```r
dtm <- DocumentTermMatrix(corpusHeadline)
sparse <- removeSparseTerms(dtm, 0.99)
headlineWords <- as.data.frame(as.matrix(sparse))
```

Let's make sure our variable names are okay for R.  


```r
colnames(headlineWords) <- make.names(colnames(headlineWords))
```

## Training the models  

First we need to split the observations back into the training set and testing 
set. To do this, we can use the `head` and `tail` functions in `R`.  


```r
headlineWordsTrain <- head(headlineWords, nrow(newsTrain))
headlineWordsTest <- tail(headlineWords, nrow(newsTest))
```

Note that this split of HeadlineWords works to properly put the observations 
back into the training and testing sets, because of how we combined them 
together when we first made our corpus.  

Before building models, we want to add back the original variables from our
datasets. We'll add back the dependent variable to the training set, and the
`WordCount` variable to both datasets. Later on we will experiment with adding
more variables to use in our model.  


```r
headlineWordsTrain$Popular <- newsTrain$Popular
headlineWordsTrain$WordCount <- newsTrain$WordCount
headlineWordsTest$WordCount <- newsTest$WordCount
```

### Logistic regression  

Now let's create a logistic regression model using all of the variables.  


```r
headlineWordsLog <- glm(Popular ~ ., data=headlineWordsTrain, family=binomial)
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

And make predictions on our test set.  


```r
headlineWordsLogPredTest <- predict(headlineWordsLog, newdata=headlineWordsTest, type="response")
```

#### Evaluation  

In order to calculate the `AUC` we need the test data to be labeled with the final
response. But we don't have those labels. Actually we cannot evaluate any model
without those labels on the test data. In order to have some feedback from our
models performance we will split our training data into train and test and 
retrain/test the models using this new split.  


```r
spl <- sample.split(headlineWordsTrain$Popular, .7)
evalHeadlineWordsTrain <- subset(headlineWordsTrain, spl==T)
evalHeadlineWordsTest <- subset(headlineWordsTrain, spl==F)
```

Train the model.  


```r
evalHeadlineLogModel <- glm(Popular~., data=evalHeadlineWordsTrain, family=binomial)
```

```
## Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred
```

Make predictions.  


```r
evalHeadlineLogPred <- predict(evalHeadlineLogModel, 
                               newdata=evalHeadlineWordsTest, type="response")
```

Get the `AUC` value.  


```r
headlineWordsLogROCR <- prediction(evalHeadlineLogPred, evalHeadlineWordsTest$Popular)
headlineWordsLogAUC <- as.numeric(performance(headlineWordsLogROCR, "auc")@y.values)
headlineWordsLogAUC
```

```
## [1] 0.7782031
```

We obtain an `AUC` value slightly higher than that we get when submitting data
to Kaggle. We need to keep this in mind. But remember, *when making submissions
we will always use the complete training data to train our models and generate
the submission file*.  

#### Generating submission file  

Now we can prepare our submission file for Kaggle.  


```r
mySubmission = data.frame(UniqueID = newsTest$UniqueID, Probability1 = headlineWordsLogPredTest)
write.csv(mySubmission, "SubmissionHeadlineLog.csv", row.names=FALSE)
```

### CART  

As an alternative that will also give us some insight into the nature of the 
data, we will build now classification trees. The corpus and training/test
data are the same. We just need to train a different type model.  


```r
headlineWordsCARTModel <- rpart(Popular~., data=evalHeadlineWordsTrain, method="class")
prp(headlineWordsCARTModel)
```

![](README_files/figure-html/unnamed-chunk-15-1.png) 

Not a very effective model. Let us get just probabilities in order to see what
is happening.  


```r
headlineWordsCARTModel <- rpart(Popular~., data=evalHeadlineWordsTrain)
prp(headlineWordsCARTModel)
```

![](README_files/figure-html/unnamed-chunk-16-1.png) 

We see the problem. Probabilities are actually too small (lower than .5) so
the tree never predicts an article as popular using that threshold.  

Let's try now to make the tree more complex.  


```r
headlineWordsCARTModel <- rpart(Popular~., data=evalHeadlineWordsTrain, cp=0.0025)
prp(headlineWordsCARTModel)
```

![](README_files/figure-html/unnamed-chunk-17-1.png) 

Let's obtain predictions using this last model.  


```r
headlineWordsCARTPred <- predict(headlineWordsCARTModel, newdata=evalHeadlineWordsTest)
```

And let's evaluate this last model.  


```r
headlineWordsCARTROCR <- prediction(headlineWordsCARTPred, 
                                   evalHeadlineWordsTest$Popular)
headlineWordsCARTauc <- as.numeric(performance(headlineWordsCARTROCR, "auc")@y.values)
headlineWordsCARTauc
```

```
## [1] 0.7411572
```

It doesn't improve the logistic regression model.  

### Random forest  

As a third method, we will try random forests.  


```r
healineWordsRF <- randomForest(Popular~., data=evalHeadlineWordsTrain)
```

```
## Warning in randomForest.default(m, y, ...): The response has five or fewer
## unique values.  Are you sure you want to do regression?
```

Make predictions.  


```r
headlineWordsRFPred <- predict(healineWordsRF, newdata=evalHeadlineWordsTest)
```

Right into `auc` calculation.  


```r
headlineWordsRFrocr <- prediction(headlineWordsRFPred, evalHeadlineWordsTest$Popular)
headlineWordsRFauc <- as.numeric(performance(headlineWordsRFrocr, "auc")@y.values)
headlineWordsRFauc
```

```
## [1] 0.7802711
```

#### Generating submission file  

This is the best model so far. Let's build it using the complete training set.  


```r
healineWordsRF <- randomForest(Popular~., data=headlineWordsTrain)
```

```
## Warning in randomForest.default(m, y, ...): The response has five or fewer
## unique values.  Are you sure you want to do regression?
```

Make predictions.  


```r
headlineWordsRFPred <- predict(healineWordsRF, newdata=headlineWordsTest)
```

Now we can prepare our submission file for Kaggle.  


```r
mySubmission <- data.frame(
    UniqueID = newsTest$UniqueID, 
    Probability1 = headlineWordsRFPred
    )
write.csv(mySubmission, "SubmissionHeadlineRF.csv", row.names=FALSE)
```



# A richer bag of words model  

# Cluster helps us understand data  

# Either cluster then predict (multiple predictors), or use most frequent terms from each cluster (term weighting)  

# Experiment: using separate models for each section and average them manually with different weights   

Rational behind this?  



