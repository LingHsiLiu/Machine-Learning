# load libeary and data 
setwd("~/ws/CS498-AML/WK-3-HW")

library("data.table")

set.seed(42)
#data processing 
adult_data <- data.table(read.csv(file="adult.csv", header=FALSE, sep=","))
continous_adult_data <- adult_data[, .(V1, V3, V5, V11, V12, V13, V15)]
continous_adult_data$V15 <- as.character(continous_adult_data$V15)
continous_adult_data$V15[continous_adult_data$V15 == " >50K"] <- 1 # change to 1 or -1
continous_adult_data$V15[continous_adult_data$V15 == " <=50K"] <- -1

adult_test_data <- data.table(read.csv(file="adult_test.csv", header=FALSE, sep=","))
continous_adult_test_data <- adult_test_data[, .(V1, V3, V5, V11, V12, V13, V15)]
continous_adult_test_data$V15 <- as.character(continous_adult_test_data$V15)
continous_adult_test_data$V15[continous_adult_test_data$V15 == " >50K."] <- 1 #change to 1 or -1
continous_adult_test_data$V15[continous_adult_test_data$V15 == " <=50K."] <- -1

my_data <- rbind(continous_adult_data, continous_adult_test_data)

cols <- c("V1", "V3", "V5", "V11", "V12", "V13")
my_data[, (cols) := lapply(list(V1, V3, V5, V11, V12, V13), scale)]

my_validation <- my_data[sample(nrow(my_data), round(0.1*nrow(my_data))), ] #4884 is 10% of total 48842

my_test <-  my_data[sample(nrow(my_data), round(0.1*nrow(my_data))), ]

my_train <- my_data[sample(nrow(my_data), round(0.8*nrow(my_data))), ] #39074 is 80% of total 48842

#evaluate function y(k)=ax(k)+b
evaluateY_K <- function(x, a, b){
  newX_K <- as.numeric(as.matrix(x))
  return (t(a) %*% newX_K + b) 
}

#magnitude of the coefficient vector 
magnitude <- function(a) {
  matrix_a <- as.numeric(as.matrix(a))
  return(t(a) %*% matrix_a)
}

#determine class label based on whether y >= 0 
convertpred <- function(Yvalue){
  if(Yvalue >= 0){
    return(1)
  }
  else{
    return(-1)
  }
}

#compute accuracy
accuracy <- function(x,y,a,b){
  predictright <- 0
  predictwrong <- 0
  for (i in 1:nrow(y)){
    predict_y <- evaluateY_K(x[i,], a, b)
    predict <- convertpred(predict_y)
    actual <- as.numeric(y[i])
    
    if(predict == actual){
      predictright <- predictright + 1 
    } else{
      predictwrong <- predictwrong + 1
    }
  }
  return(predictright / (predictright+predictwrong))
}

validation_accuracies = c()
test_accuracies = c()
list_accuracies <- list()
list_magnitudes <- list()
i <-  0
lambdas <- c(.001, .01, .1, 1)

for (lambda in lambdas){
  #intialize a and b with random numbers
  a <- runif(6, min=-.0001, max=.0001)
  b <- runif(1, min=0, max=.01)
  
  accuracies <- c()
  magnitudes <- c()
  
  # 50 epochs
  for (epoch in 1:50){
    
    #set out 50 examples for testing 
    held_out_rows <- sample(1:dim(my_train)[1], 50)
    eva_data <- my_train[held_out_rows, ][, .(V1, V3, V5, V11, V12, V13)]
    eva_labels <- my_train[held_out_rows, ][, .(V15)]
    train_data <- my_train[-held_out_rows,][, .(V1, V3, V5, V11, V12, V13)]
    train_labels <- my_train[-held_out_rows][, .(V15)]
    
    #count step
    numberSteps <- 0
    
    #300 steps
    for (step in 1:300){
      
      #if do 30 steps to generate an accuracy, then use 50 sample to do testing
      if(numberSteps %% 30 == 0){
        # 50 samples to do testing
        accuracy_calculate <- accuracy(eva_data, eva_labels, a, b)
        magn <- magnitude(a)
        magnitudes <- c(magnitudes, magn)
        accuracies <- c(accuracies, accuracy_calculate) 
      }
      
      #select random k from other training 
      k <- sample(1:nrow(train_labels), 1)
     
      
      #get x(k) and y(k) value
      exampleX <- as.numeric(as.matrix( train_data[k,] ))
      exampleY <- as.numeric(train_labels[k])
      
      #get value y(k) = ax+b
      predict <- evaluateY_K(exampleX, a, b)
      
      #gradient vector 
      #a(n+1)=a(n)-steplength*(lambda*a), if y(k) * (a (T) X(k)) >= 1
      #a(n+1)=a(n)-steplength*(lambda*a - y(k)X), otherwise 
      #b(n+1)=b(n)-steplength*0, if y(k) * (a (T) X(k)) >= 1
      #b(n+1)=b(n)-steplength*(-y(k)), otherwise
      
      if(exampleY * predict >= 1){
        newa <- lambda * a
        newb <- 0
      } else {
        newa <- (lambda * a) - (exampleY * exampleX)
        newb <- -(exampleY)
      }
      
      #gradient descent η(e) = m / e+n, m and n is contant, from page 35
      steplength = 1/(0.01 * epoch + 50)
      
      #a and b after gradient descent
      a <- a - (steplength * newa)
      b <- b - (steplength * newb)      
      
      #steps count
      numberSteps <- numberSteps + 1
    }
  }
  
  validation_evaluate <- accuracy (my_validation[, .(V1, V3, V5, V11, V12, V13)], my_validation[, .(V15)], a, b)

  validation_accuracies <- c(validation_accuracies, validation_evaluate)
  
  test_evaluate <- accuracy(my_test[, .(V1, V3, V5, V11, V12, V13)], my_test[, .(V15)], a, b)
  
  test_accuracies <- c(test_accuracies, test_evaluate)
  
  #collect accuracies for each lambda and store them in list for plotting
  i <- i + 1
  list_accuracies[[i]] <- accuracies
  list_magnitudes[[i]] <- magnitudes
}

#plot
  cl <- rainbow(4)
  jpeg(file="Accuracy.jpg")
  title <- "Accuracy Graph"
  plot(1:length(list_accuracies[[1]]) , list_accuracies[[1]], type="l", col=cl[1], xlab ="Epoch", ylab ="Accuracy", main = title)
  for (i in 2:4)
    lines(1:length(list_accuracies[[i]]) , list_accuracies[[i]], type="l", col=cl[i])
  legend("bottomright", legend=lambdas, col=cl, lty=1:2, cex=0.8)
  dev.off()
  
  jpeg(file="Magnitude.jpg")
  title <- "Magnitude Graph"
  plot(1:length(list_magnitudes[[1]]) , list_magnitudes[[1]], type="l", col=cl[1], xlab ="Epoch", ylab ="Magnitude", main = title)
  for (i in 2:4)
    lines(1:length(list_magnitudes[[i]]) , list_magnitudes[[i]], type="l", col=cl[i])
  legend("topleft", legend=lambdas, col=cl, lty=1:2, cex=0.8)
  dev.off()

#find the best regularization constant (lambda)
index_max <- 1
for(i in 1:length(validation_accuracies)){
  if (validation_accuracies[i] >= validation_accuracies[index_max]){
    index_max <- i
  }
}
lambda_max <- lambdas[index_max]

#accuracy of the best classifier
best_accuracy <- test_accuracies[index_max]


