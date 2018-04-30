# load libeary and data 
library(caret)
setwd('~/Desktop/CSAML')
wdat<-read.csv('pima-indians-diabetes.data.txt', header=FALSE)

# data preprocessing
X<-wdat[,-c(9)]
Y<-wdat[,9]

#random split ten times
testscore<-array(dim=10)
for(i in 1:10){
  # splite data into train data and test data
  datas<-createDataPartition(y=Y, p=.8, list=FALSE)
  trainX<-X[datas,]
  trainY<-Y[datas]
  testX<-X[-datas,]
  testY<-Y[-datas]
  
  #splite data into positive & negative class
  pflag<-trainY>0 # TRUE & FALSE
  p<-trainX[pflag,]
  n<-trainX[!pflag,]
  
  # calculate mean and standard deviation
  pmean<-sapply(p, mean, na.rm=TRUE)
  nmean<-sapply(n, mean, na.rm=TRUE)
  psd<-sapply(p, sd, na.rm=TRUE)
  nsd<-sapply(n, sd, na.rm=TRUE)
  
  #calculate probability for positive & negative class
  pprobability1<-t(t(testX)-pmean)
  pprobability2<-t(t(pprobability1)/psd)
  
  nprobability1<-t(t(testX)-nmean)
  nprobability2<-t(t(nprobability1)/nsd)
  
  # +log(+) #c(1,2) means rows and columns(meaning everything in the matrix)
  ptemaxlike<--(1/2)*rowSums(apply(pprobability2,c(1,2), function(x)x^2),na.rm=TRUE)-sum(log(psd))+log(nrow(p)/(nrow(p)+nrow(n)))
  
  
  # +log(-)
  ntemaxlike<--(1/2)*rowSums(apply(nprobability2,c(1,2), function(x)x^2),na.rm=TRUE)-sum(log(nsd))+log(nrow(n)/(nrow(p)+nrow(n)))
  
  
  #pick the maximum likelyhood and check the answer and record score
  predict<-ptemaxlike>ntemaxlike
  predictright<-predict==testY
  testscore[i]<-sum(predictright)/(sum(predictright)+sum(!predictright))
}

accuracy <- sum(testscore) / length(testscore)
#average accuracy after 10 times cross validate
accuracy

#print 10 timestestscore
print(paste0("score: ", testscore))
