library(caret)
library(tidyverse)
library(randomForest)
library(data.table)
library(dplyr)

Data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/movies-mld/"),
                 header = F, sep = ",")
Train <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn"),
                  header = F, sep = "")
Test <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst"),
                  header = F, sep = "")
#==============================================================================#
# Data preparation
#==============================================================================#

# Fix the names of the data if needed
Data <- Data %>%
  rename(Name = V1,
         mcg = V2,
         gvh = V3,
         alm = V4,
         mit = V5,
         erl = V6,
         pox = V7,
         vac = V8,
         nuc = V9,
         location = V10)

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Turn the outcome varialbe into a factor for classification
Data$Class <- as.factor(Data$Class)

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$Class, 
                                  p = .8, #(80/20 split of the data)
                                  list = FALSE, 
                                  times = 1)

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#Validate split
#You want the outcome variable to have an equal split in the train vs test groups
count_train <- table(Train$V35)
count_train["g"]/sum(count_train)

count_test <- table(Test$V35)
count_test["g"]/sum(count_test)

#==============================================================================#
# Random forest 
#==============================================================================#
Train$V37 <- as.factor(Train$V37)

set.seed(24)
rf <- randomForest(V37 ~ .,
                   data = Train,
                   mtry = 6, #For classificantion mtry = sqrt(column amount), regression = column amount / 3
                   ntree = 3001, #make odd 
                   importance = TRUE)

#==============================================================================#
# Determine when the error rates level off
#==============================================================================#

OOB.error.rates <- data.frame(Trees = rep(1:nrow(rf$err.rate), times = 7),
                              Type = rep(c("OOB", "1", "2", "3", "4", "5", "7"), 
                                         each=nrow(rf$err.rate)),
                              Error = c(rf$err.rate[,"OOB"],
                                        rf$err.rate[,"1"],
                                        rf$err.rate[,"2"],
                                        rf$err.rate[,"3"],
                                        rf$err.rate[,"4"],
                                        rf$err.rate[,"5"],
                                        rf$err.rate[,"7"]))

ggplot(data = OOB.error.rates, aes(x = Trees, y = Error)) +
  geom_line(aes(color= Type))

# If it does not look like it has leveled off go back and add more trees to test

#==============================================================================#
# Tune mtry
#==============================================================================#

set.seed(24)

oob.values <- vector(length = 10)
for(i in 1:10){
  temp.model <- randomForest(V37 ~ .,
                             data = Train,
                             mtry = i,
                             ntree = 3001)
  
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate), 1]
}

# Get the location of the mtry with the smallest error
which.min(oob.values)

#==============================================================================#
# Run the now tuned randomForest model
#==============================================================================#

set.seed(24)
rf1 <- randomForest(V37 ~ .,
                    data = Train,
                    mtry = 8, #from tuneRF
                    ntree = 3001, #from tuneRF
                    importance = TRUE,
                    proximity = TRUE)

#==============================================================================#
# How does the model perform on the test data
#==============================================================================#

test_prediction <-predict(rf1, Test)

# for classification models
confusionMatrix(test_prediction, as.factor(Test$V37))

# RMSE for regression models
rmse = caret::RMSE(Test$V37, xgb.pred)
#==============================================================================#
# What variables contributed the most to the model
#==============================================================================#

varImpPlot(rf1)


