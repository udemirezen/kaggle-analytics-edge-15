set.seed(1234)
rfModel <- randomForest(Popular~.-UniqueID, data=evalNewsTrain, ntree=10000)
rfPred <- predict(rfModel, newdata=evalNewsTest)

# Calculate accuracy
table(evalNewsTest$Popular, rfPred>.5)

# Check test cases where our prediction failed
errors <- newsTest[evalNewsTest$Popular!=(rfPred>.5),]
summary(errors)
par(mfrow=c(3,2))
plot(errors$NewsDesk)
plot(evalNewsTest$NewsDesk)
plot(errors$Section)
plot(evalNewsTest$Section)
plot(errors$Subsection)
plot(evalNewsTest$Subsection)

# Calculate AUC
rfRocr <- prediction(rfPred, evalNewsTest$Popular)
rfAuc <- as.numeric(performance(rfRocr, "auc")@y.values)
rfAuc

# Save to file
set.seed(1234)
rfModelSubmission <- randomForest(Popular~.-UniqueID, data=newsTrain, ntree=10000)
rfPredSubmission <- predict(rfModelSubmission, newdata=newsTest)
mySubmission <- data.frame(
    UniqueID = newsTest$UniqueID, 
    Probability1 = abs(rfPredSubmission)
)
write.csv(mySubmission, "SubmissionRF_20diff_nt10000.csv", row.names=FALSE)
