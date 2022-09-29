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

#Data <- read.csv(url(), header = FALSE)
Data <- fread("C:\\Users\\owner\\Documents\\Thesis_Documents\\Sheps_temp.csv")

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

# Fix the names of the data if needed
Data <- Data %>%
  mutate(month = month(Date),
         Outcome = fifelse(Respiratory == 1, 1,
                     fifelse(Cardiovascular == 1, 2,
                       fifelse(died == 1, 3,
                         fifelse(Sui == 1, 4,
                           fifelse(Selfharm == 1, 5,
                             fifelse(Mental_Health == 1, 6, 0))))))) %>%
  filter(month >= 6, month <= 9, sex != "U") %>% #only looking at the months June thru September
  relocate(Mental_Health, .after = Outcome) %>%
  dplyr::select(-Date, -source, -shepsid, -Respiratory, -Cardiovascular, -died, -Sui, -Selfharm, -Mood, -Outcome)


# Change the values of the prediction column from character to numeric
Data$sex <- fifelse(Data$sex == "F", 1, 2)

# Turn characters to numerics
Data <- mutate_all(Data, function(x) as.numeric(as.character(x)))

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$Mental_Health, 
                                  p = .8, #(80/20 split of the data)
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#==============================================================================#
# One hot encoding Train features
#==============================================================================#

lab <- Train[16]
dummy <- dummyVars(" ~ .", 
                   data = Train[,-16])
newdata <- data.frame(predict(dummy, newdata = Train[,-16]))
Train <- cbind(newdata, lab)

colnames(Train)[length(Train)] <- "Mental_Health"

# Remove extra columns that dont appear in the Test
#Train <- Train[,-c(17)]

# One hot encoding Test features

lab_test <- Test[,16]
dummy <- dummyVars(" ~ .", 
                   data = Test[,-16])
newdata <- data.frame(predict(dummy, newdata = Test[,-16]))
Test <- cbind(newdata, lab_test)

colnames(Test)[length(Test)] <- "Mental_Health"

#==============================================================================#
# Build model
#==============================================================================#

grid_tune <- expand.grid(nrounds = c(500,1000,1500), #number of trees
                         max_depth = c(2,4,6, 8), 
                         eta = c(0.025, 0.05, 0.1, 0.3, 0.5), #Learning rate
                         gamma = c(0, 0.1, 0.5), #c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0), #pruning --> should be tuned c(0, 0.05, 0.1, 0.5, 0.7, 0.9, 1.0)
                         colsample_bytree = c(0.8, 1.0), # subsample ratio of columns for tree
                         min_child_weight = 1, #c(1,2,3), # the larger, the more conservative the model
                         subsample = 1) # c(0.5, 0.75, 1.0) # used to prevent overfitting by sampling x% training

train_control <- trainControl(method = "cv",
                              number = 10,
                              verboseIter = TRUE,
                              allowParallel = TRUE)

# This sets up the parallel computing ability
registerDoParallel(cl)

set.seed(24)
xgb_tune <- caret::train(x = Train[,-16],
                         y = Train[, 16],
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
xgb_model <- caret::train(x = Train[,-16],
                          y = Train[, 16],
                          trControl = train_control,
                          tuneGrid = grid_tune,
                          method = "xgbTree",
                          verbose = TRUE)

#==============================================================================#
# Model evaluation
#==============================================================================#

# Prediction
xgb.pred <- predict(xgb_model, Test)

# RMSE for regression models
rmse = caret::RMSE(Test$Class, xgb.pred)
