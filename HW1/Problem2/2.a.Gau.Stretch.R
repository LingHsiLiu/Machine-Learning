library(e1071)
library(klaR)
library(caret)
library(imager)

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
}

# load label files
load_label_file = function(filename) {
  f = file(filename, 'rb')
  readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  n = readBin(f, 'integer', n = 1, size = 4, endian = 'big')
  y = readBin(f, 'integer', n = n, size = 1, signed = FALSE)
  close(f)
  y
}

trainx = load_image_file("/Users/Althea/Desktop/CSAML/Problem2/train-images-idx3-ubyte")
testx <<- load_image_file("/Users/Althea/Desktop/CSAML/Problem2/t10k-images-idx3-ubyte")

trainy <<- load_label_file("/Users/Althea/Desktop/CSAML/Problem2/train-labels-idx1-ubyte")
testy <<- load_label_file("/Users/Althea/Desktop/CSAML/Problem2/t10k-labels-idx1-ubyte")

#data processing
trainLabels <- as.factor(trainy)

#doing thresholding for train data
trainingx <- trainx>127
trainingx[trainingx == FALSE] <- 0    
trainingx[trainingx == TRUE] <- 1

#doing thresholding for test data
testingx <- testx>127
testingx[testingx == FALSE] <- 0    
testingx[testingx == TRUE] <- 1

#resized to a smaller for train data
scaled.trainx <- apply(trainingx, 1, function(x) resize(autocrop(as.cimg(x)), 20, 20))
scaled.trainx <- t(scaled.trainx)

#resized to a smaller for test data
scaled.testx <- apply(testingx, 1, function(x) resize(autocrop(as.cimg(x)), 20, 20))
scaled.testx <- t(scaled.testx)

# train naiveBayes model
model <- naiveBayes(scaled.trainx, trainLabels)

# using model to do testing
prediction <- predict(model, newdata = scaled.testx)
confusionMatrix(data = testy, prediction)