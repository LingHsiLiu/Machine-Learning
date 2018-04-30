library(e1071)
library(klaR)
library(caret)
library(h2o)
library(randomForest)

# load image files
load_image_file = function(filename) {
  ret = list()
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n    = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  nrow = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  ncol = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  x = readBin(f, 'integer', n = n * nrow * ncol, size = 1, signed = FALSE)
  close(f)
  matrix(x, ncol = nrow * ncol, byrow = TRUE)
  #data.frame(matrix(x, ncol = nrow * ncol, byrow = TRUE))
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  #y
  #data.frame(matrix(y, ncol = n, byrow = TRUE))
  matrix(y, ncol = n, byrow = TRUE)
}

trainx = load_image_file("/Users/Althea/Desktop/CSAML/Problem2/train-images-idx3-ubyte")
testx <<- load_image_file("/Users/Althea/Desktop/CSAML/Problem2/t10k-images-idx3-ubyte")

trainy <<- load_label_file("/Users/Althea/Desktop/CSAML/Problem2/train-labels-idx1-ubyte")
testy <<- load_label_file("/Users/Althea/Desktop/CSAML/Problem2/t10k-labels-idx1-ubyte")

#doing cropped and resize
trainingx <- trainx>127
trainingx[trainingx == FALSE] <- 0    
trainingx[trainingx == TRUE] <- 1

testingx <- testx>127
testingx[testingx == FALSE] <- 0    
testingx[testingx == TRUE] <- 1

scaled.trainx <- apply(trainingx, 1, function(x) resize(autocrop(as.cimg(x)), 20, 20))
scaled.trainx <- t(scaled.trainx)

scaled.testx <- apply(testingx, 1, function(x) resize(autocrop(as.cimg(x)), 20, 20))
scaled.testx <- t(scaled.testx)

#initial h2o
h2o.init(nthreads=-1)
#trasfer data from data frame to h2o frame
trainH2o <- as.h2o(data.frame(x = scaled.trainx, y = as.factor(trainy)))
testH2o <- as.h2o(data.frame(x = scaled.testx, y = as.factor(testy)))

#create model and calculate accuracy
#use ntrees to adjust tree parameter 10, 20 ,30
#use max_depth to adjust depth 4, 8, 16
model = h2o.randomForest(training_frame = trainH2o, y = 'y', ntrees = 10, max_depth = 16)
prediction.forest <- h2o.predict(model, testH2o) 
mean(as.data.frame(prediction.forest)[, 1] == testy)