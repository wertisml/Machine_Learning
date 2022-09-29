library(caret)
library(tidyverse)
library(data.table)
library(dplyr)
library(MASS)
library(xgboost)
library(magrittr)
library(Matrix)
library(mltools)
library(doParallel)
library(DMwR)
library(unbalanced)
data(iris)
Data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"),
                 header = FALSE)
Data <- iris
#==============================================================================#
# Set up parallel
#==============================================================================#

# Sets the number of cores to be used but makes sure that there is 1 core left free
cl <- makePSOCKcluster(detectCores() - 1)

unregister <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

#==============================================================================#
# Data preparation
#==============================================================================#

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Change the values of the prediction column from character to numeric
Data$Species <- fifelse(Data$Species == "setosa", 1,
                        fifelse(Data$Species == "versicolor", 2, 3))

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$Class, 
                                  p = .8, #(80/20 split of the data)
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#==============================================================================#
# One hot encoding Train features (For Classification)
#==============================================================================#

lab <- Train[11]
dummy <- dummyVars(" ~ .", 
                   data = Train[,-11])
newdata <- data.frame(predict(dummy, newdata = Train[,-11]))
Train <- cbind(newdata, lab)

colnames(Train)[length(Train)] <- "Class"

# Remove column 82 since the test data does not have native_country_holland
#Train <- Train[,-c(17)]
Train$Class <- as.factor(Train$Class)

# One hot encoding Test features

#Test$'|1x3 Cross validator' <- as.numeric(Test$'|1x3 Cross validator')
lab_test <- Test[,11]
dummy <- dummyVars(" ~ .", 
                   data = Test[,-11])
newdata <- data.frame(predict(dummy, newdata = Test[,-11]))
Test <- cbind(newdata, lab_test)

colnames(Test)[length(Test)] <- "Class"

Test$Class <- as.factor(Test$Class)

#==============================================================================#
# Build model
#==============================================================================#

grid_tune <- expand.grid(nrounds = c(500,1000,1500), #number of trees
                         max_depth = c(2,4,6), 
                         eta = 0.05, #c(0.025, 0.05, 0.1, 0.3, 0.5), #Learning rate
                         gamma = 0.5, #c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0), #pruning --> should be tuned c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0)
                         colsample_bytree = c(0.8, 1.0), # subsample ratio of columns for tree
                         min_child_weight = 1, #c(1,2,3), # the larger, the more conservative the model
                         subsample = 1) # c(0.5, 0.75, 1.0) # used to prevent overfitting by sampling x% training

train_control <- trainControl(method = "cv",
                              number = 10,
                              sampling = "smote", #if there is an imbalance in the classes
                              verboseIter = TRUE,
                              allowParallel = TRUE)

# This sets up the parallel computing ability
registerDoParallel(cl)

set.seed(24)
xgb_tune <- caret::train(x = Train[,-21],
                  y = Train[, 21],
                  trControl = train_control,
                  tuneGrid = grid_tune,
                  method = "xgbTree",
                  verbose = TRUE)

# This turns off the parallel
stopCluster(cl)
unregister()

#==============================================================================#
# Run the tuned model
#==============================================================================#

grid_tune <- xgb_tune$bestTune

train_control <- trainControl(method = "cv",
                              number = 10,
                              verboseIter = TRUE)


set.seed(24)
xgb_model <- caret::train(x = Train[,-21],
                   y = Train[, 21],
                   trControl = train_control,
                   tuneGrid = grid_tune,
                   method = "xgbTree",
                   verbose = TRUE)

#==============================================================================#
# Model evaluation
#==============================================================================#

# Prediction
xgb.pred <- predict(xgb_model, Test)

# Confusion Matrix
confusionMatrix(as.factor(as.numeric(xgb.pred)),
                as.factor(as.numeric(Test$Class)),
                mode = "everything")
