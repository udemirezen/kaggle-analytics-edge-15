set.seed(1234)
logModel <- glm(Popular~.-UniqueID, data=evalNewsTrain, family=binomial)
summary(logModel)

logPred <- predict(logModel, newdata=evalNewsTest, type="response")

# Calculate accuracy
table(evalNewsTest$Popular, rfPred>.5)
