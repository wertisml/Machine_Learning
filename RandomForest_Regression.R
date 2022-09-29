library(ranger)
library(caret)
library(tidyverse)
library(randomForest)
library(data.table)
library(dplyr)
library(DMwR)
library(unbalanced)

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

# Turn characters to numerics
Data <- mutate_all(Data, function(x) as.numeric(as.character(x)))

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$Class, 
                                  p = .8, #(80/20 split of the data)
                                  list = FALSE, 
                                  times = 1)

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#==============================================================================#
# Random forest 
#==============================================================================#

# Tune the best tree fit
nt <- seq(1, 5001, 250)

oob_mse <- vector("numeric", length(nt))

set.seed(24)

for(i in 1:length(nt)){
  fit <- ranger(Class ~ ., 
                data = Train, 
                num.trees = nt[i],
                mtry = 6,
                #max.depth = 8,
                #probability = TRUE,
                classification = FALSE)
  oob_mse[i] <- fit$prediction.error
}


plot(x = nt, y = oob_mse, col = "red", type = "l")

ntree <- (which.min(oob_mse)*250)+1

# Tune the best mtry
set.seed(24)
oob.values <- vector(length = 10)
for(i in 1:10){
  temp.model <- ranger(V37 ~ .,
                       data = Train,
                       mtry = i,
                       num.trees = ntree)
  
  oob.values[i] <- temp.model$prediction.error
}

# Get the location of the mtry with the smallest error
mtry <- which.min(oob.values)

#==============================================================================#
# Run the now tuned randomForest model
#==============================================================================#

set.seed(24)
rf1 <- ranger(V37 ~ .,
                    data = Train,
                    mtry = mtry, #from tuneRF
                    num.trees = ntree, #from tuneRF
                    classification = FALSE)

#==============================================================================#
# How does the model perform on the test data
#==============================================================================#

probabilities = predict(rf1, data = Test)$predictions
rmse = caret::RMSE(Test$V37, probabilities)
