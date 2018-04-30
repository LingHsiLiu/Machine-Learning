# load libeary and data 
library(caret)
setwd('~/Desktop/CSAML')
wdat<-read.csv('pima-indians-diabetes.data.txt', header=FALSE)

# data preprocessing
X<-wdat[,-c(9)]
Y<-wdat[,9]

# splite data into train data and test data
datas<-createDataPartition(y=Y, p=.8, list=FALSE)
trainX<-X[datas,]
trainY<-Y[datas]
testX<-X[-datas,]
testY<-Y[-datas]

#using svmlight to train svm
svm <- svmlight(trainX, factor(trainY), pathsvm="/Users/Althea/Desktop/CSAML/svm_light_osx.8.4_i7/")

#predict
labels <- predict(svm, testX)
predict <- labels$class

#accuracy
predictright <- sum(predict == testY)
predictwrong <- sum(predict != testY)
accuracy <- predictright / (predictright + predictwrong)
accuracy
