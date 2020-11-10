#Install and load required packages
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
#sudo apt install libtiff-dev fftw-dev libfftw3-bin libfftw3-dev libmagick++-dev ffmpeg
if(!require(imager)) install.packages("imager", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(doParallel)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(purrr)) install.packages("doParallel", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("gbm", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("MASS", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("plyr", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(imager)
library(matrixStats)
library(randomForest)
library(doParallel)
library(purrr)
library(gbm)
library(naivebayes)
library(MASS)
library(plyr)
library(knitr)

#Set working directory for Windows, not needed for Linux
#setwd("E:/Documents/edX/PH125-9x - Data Science - Capstone/casting-defect")

#initial trials with importing images
image <- load.image("data/test/ok_front/cast_ok_0_10.jpeg")
dim(image)
image
plot(image, axes = false)
grey_image <- grayscale(image)
grey_image_small <- imresize(grey_image,scale=0.10)
image_array <- as.numeric(grey_image_small)

#Image re-size factor
#start with 10% to minimize size of data
scale=0.10

#function to open all jpeg images in directory, change to greyscale, and resize
#turn all images into a row in a matrix with values 0 to 1
get_image_matrix <- function(path,scale){
    files <- list.files(path=path, pattern=".jpeg",all.files=T, full.names=T, no.. = T) 
    list_of_images <- map(files, function(.x){
      image <- load.image(.x)
      grey_image <- grayscale(image)
      grey_image_small <- imresize(grey_image,scale=scale)
    })
    image_matrix = do.call('rbind', map(list_of_images, as.numeric))
    return(image_matrix)
}

#import test data
#images within folder "ok_front" labelled as "pass"
x_test_pass <- get_image_matrix(path="data/test/ok_front",scale=scale)
y_test_pass <- array(data=rep.int("pass",nrow(x_test_pass)))
#images within folder "def_front" labelled as "fail"
x_test_fail <- get_image_matrix(path="data/test/def_front",scale=scale)
y_test_fail <- array(data=rep.int("fail",nrow(x_test_fail)))

#combine pass and fail test data into single dataset
x_test <- rbind(x_test_pass,x_test_fail)
y_test <- factor(array(c(y_test_pass,y_test_fail)))

#import train data
#images within folder "ok_front" labelled as "pass"
x_train_pass <- get_image_matrix(path="data/train/ok_front",scale=scale)
y_train_pass <- array(data=rep.int("pass",nrow(x_train_pass)))
#images within folder "def_front" labelled as "fail"
x_train_fail <- get_image_matrix(path="data/train/def_front",scale=scale)
y_train_fail <- array(data=rep.int("fail",nrow(x_train_fail)))

#combine pass and fail train data into single dataset
x_train <- rbind(x_train_pass,x_train_fail)
y_train <- factor(array(c(y_train_pass,y_train_fail)))

#dimension check of variables for errors
dim(x_train_pass)
dim(x_train_fail)
dim(x_train)
str(x_train)

length(y_train)
str(y_train)
table(y_train)

#Replot images to ensure no errors
image <- load.image("data/test/ok_front/cast_ok_0_10.jpeg")
image_dim <- sqrt(ncol(x_train_pass))
new_image <- matrix(data=x_test_pass[1,],nrow=image_dim,ncol=image_dim,byrow=TRUE)
new_image <- t(new_image)

par(mar = rep(0, 4),pty="s")
image(new_image[,nrow(new_image):1], axes = FALSE, col = grey(seq(0, 1, length = 256)))
plot(image, axes=FALSE)



######################### Pre-processing #########################
#Check columns for zero variability
sds <- colSds(x_train)
qplot(sds, bins = 256, color = I("black"))

#caret function for near zero variance
nzv <- nearZeroVar(x_train)
image(matrix(1:ncol(x_train) %in% nzv, image_dim, image_dim))
col_index <- setdiff(1:ncol(x_train), nzv)
length(col_index)
#all columns kept - may be valuable as image scale is increased

#add column names to x_train and x_test for caret
colnames(x_train) <- 1:ncol(x_train)
colnames(x_test) <- colnames(x_train)
##################################################################

######### Fit several models and then select the best ones #######
models <- c("naive_bayes","qda","lda","gbm","knn","rf","xgbDART")
set.seed(1, sample.kind = "Rounding")
# 2-fold cross-validation to minimize computation time
control <- trainControl(method = "cv", number = 2, p = 0.8)
#use multiple CPU cores to reduce processing time
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
# map_df to loop through models
fits <- map_df(models, function(model){ 
  print(model)
  #initial timestamp
  ptm <- proc.time()
  fit<-train(x = x_train[, col_index],y = y_train, method = model, trControl = control)
  #second timestamp - find difference
  ptm <- proc.time() - ptm
  #create tibble with model name, processing time, min, max, and avg accuracy
  tibble(model=model, time=ptm[[3]], min=min(fit$results$Accuracy),max=max(fit$results$Accuracy),avg=mean(fit$results$Accuracy))
}) 
stopCluster(cl)
#output results of group of models
fits %>% arrange(desc(avg)) %>% kable()

#top four models are rf, knn, xgbDART, and gbm

##################################################################


#for parameter selection, use bootstrap with 10 iterations

###################### GBM Model ##################################
#prediction using bootstrap to determine parameters
#use multiple CPU cores to reduce processing time
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "boot", number = 10, classProbs=TRUE, summaryFunction = twoClassSummary)
tune <- expand.grid(interaction.depth=c(3,5,7), n.trees=c(100,500), shrinkage=c(0.1), n.minobsinnode=10)
ptm <- proc.time()
train_gbm <- train(x_train[, col_index], y_train,
                   method = "gbm", 
                   metric = "ROC",
                   tuneGrid = tune,
                   trControl = control)
proc.time() - ptm
stopCluster(cl)
#plot effect of parameters
ggplot(train_gbm)
train_gbm$bestTune %>% kable()

#predict test data using optimized parameters
y_hat_gbm <- predict(train_gbm, 
                     x_test[, col_index]
                     )
#since "fail" comes before "pass" in alphabet
#"fail" is positive outcome in confusionMatrix
#in a quality prediction case, high sensitivity is better than overall accuracy
cm_gbm <- confusionMatrix(data = y_hat_gbm, reference = y_test)
cm_gbm
results <- tibble(Method="GBM",
                  Sensitivity = cm_gbm$byClass["Sensitivity"],
                  F1 = cm_gbm$byClass["F1"],
                  Accuracy = cm_gbm$overall["Accuracy"]
                  )

###################################################################

###################### KNN Model ##################################
#prediction using bootstrap to determine parameters
#use multiple CPU cores to reduce processing time
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
ptm <- proc.time()
set.seed(1, sample.kind = "Rounding")
control <- trainControl(method = "boot", number = 10, classProbs=TRUE, summaryFunction = twoClassSummary)
train_knn <- train(x_train[, col_index], y_train,
                   method = "knn", 
                   metric = "ROC",
                   tuneGrid = data.frame(k = c(7,11,15)),
                   trControl = control)
proc.time() - ptm
stopCluster(cl)
#plot effect of parameters
ggplot(train_knn)
train_knn$bestTune %>% kable()

#fit model using optimized nearest neighbors
y_hat_knn <- predict(train_knn, 
                     x_test[, col_index]
)
cm_knn <- confusionMatrix(data = y_hat_knn, reference = y_test)
cm_knn
results <- bind_rows(results,
                     tibble(Method="KNN",
                            Sensitivity = cm_knn$byClass["Sensitivity"],
                            F1 = cm_knn$byClass["F1"],
                            Accuracy = cm_knn$overall["Accuracy"]
                            )
                     )
##################################################################

#################### Random Forest Model #########################
#not included in edX report for brevity
#prediction using bootstrap to determine parameters
#use multiple CPU cores to reduce processing time
cl <- makePSOCKcluster(detectCores())
registerDoParallel(cl)
ptm <- proc.time()
control <- trainControl(method = "boot", number = 5, classProbs=TRUE, summaryFunction = twoClassSummary)
grid <- data.frame(mtry = c(5, 10, 15, 20, 25))

train_rf <-  train(x_train[, col_index], y_train, 
                   method = "rf", 
                   metric = "ROC",
                   trControl = control,
                   tuneGrid = grid
                   )
proc.time() - ptm
stopCluster(cl)
#plot effect of parameters
ggplot(train_rf)
train_rf$bestTune


#fit model using optimized parameters
y_hat_rf <- predict(train_rf, 
                     x_test[, col_index]
)
cm_rf <- confusionMatrix(data = y_hat_rf, reference = y_test)
cm_rf
results <- bind_rows(results,
                     tibble(Method="Random Forest",
                            Sensitivity = cm_rf$byClass["Sensitivity"],
                            F1 = cm_rf$byClass["F1"],
                            Accuracy = cm_rf$overall["Accuracy"]
                            )
                      )

#####################################################################
results %>% kable()