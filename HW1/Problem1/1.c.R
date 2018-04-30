# load libeary and data 
library(klaR)
library(caret)
setwd('~/Desktop/CSAML')
wdat<-read.csv('pima-indians-diabetes.data.txt', header=FALSE)

# data preprocessing
X<-wdat[,-c(9)]
Y<-wdat[,9]

#create training and testing sets
testaccuracy <- array(dim=10)

for(i in 1:10){
  # splite data into train data and test data
  datas<-createDataPartition(y=Y, p=.8, list=FALSE)
  trainX<-X[datas,]
  trainY<-Y[datas]
  testX<-X[-datas,]
  testY<-Y[-datas]
  
  #using klaR package to train naive bayes model
  model <- train (trainX, factor(trainY), 'nb', trControl=trainControl(method='cv', number=10))
  
  #predict
  predict <- predict(model, newdata=testX)
  predictright <- length(testY[testY == predict])
  predictwrong <- length(testY[testY != predict])
  accuracy <- predictright / (predictright + predictwrong)
  testaccuracy[i] <- accuracy
}
accuracy_cross_validation <- sum(testaccuracy)/length(testaccuracy)

#average accuracy after 10 times cross validate
accuracy_cross_validation
#print 10 times
print(paste0("testaccuracy: ", testaccuracy))
