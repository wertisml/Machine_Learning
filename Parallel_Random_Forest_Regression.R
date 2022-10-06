library(ranger)
library(caret)
library(tidyverse)
library(randomForest)
library(data.table)
library(dplyr)
library(ggpubr)
library(vip)
library(parallel)
library(foreach)

Data <- fread("E:\\Thesis_Documents\\Sheps_temp.csv")

n.cores <- parallel::detectCores() - 1

#==============================================================================#
# Prep the data
#==============================================================================#

# Fix the names of the data if needed
Data <- Data %>%
  mutate(month = month(Date)) %>%
  filter(month >= 6, month <= 9, Date >= '2019-01-01') %>% #only looking at the months June thru September
  select(Zip, TAVG, TAGVLag1, TAGVLag2, TAGVLag3, TAGVLag4, TAGVLag5, TAGVLag6, RH, Above_95th,
         Mental_Health)

# Change the values of the prediction column from character to numeric
#Data$sex <- fifelse(Data$sex == "F", 1, 2)

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

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#==============================================================================#
#Calculating the best min.node.size
#==============================================================================#

myGrid <- expand.grid(num.trees = seq(2251, 3751, 250), #Can come back in and tighten this search window for the second round
                      mtry = 3,
                      min.node.size = 1)

#create and register cluster
my.cluster <- parallel::makeCluster(n.cores)
doParallel::registerDoParallel(cl = my.cluster)

set.seed(24)
#fitting each rf model with different hyperparameters
prediction.error <- foreach(num.trees = myGrid$num.trees,
                            mtry = myGrid$mtry,
                            min.node.size = myGrid$min.node.size,
                            .combine = 'c', 
                            .packages = "ranger") %dopar% {
                              
                              #fit model
                              m.i <- ranger::ranger(data = Train,
                                                    dependent.variable.name = "Mental_Health",
                                                    num.trees = num.trees,
                                                    mtry = mtry,
                                                    min.node.size = min.node.size)
                              
                              #returning prediction error as percentage
                              return(m.i$prediction.error * 100)
                              
                            }

parallel::stopCluster(cl = my.cluster)

#==============================================================================#
# Best ntree
#==============================================================================#

myGrid$prediction.error <- prediction.error

best.hyperparameters <- myGrid %>% 
  dplyr::arrange(prediction.error) %>% 
  dplyr::slice(1)

#==============================================================================#
# Calculating the best mtry
#==============================================================================#

myGrid <- expand.grid(num.trees = best.hyperparameters$num.trees,
                      mtry = seq(1,10,2), #Can come back in and tighten this search window for the second round
                      min.node.size = 1)

set.seed(24)
#fitting each rf model with different hyperparameters
prediction.error <- foreach(num.trees = myGrid$num.trees,
                            mtry = myGrid$mtry,
                            min.node.size = myGrid$min.node.size,
                            .combine = 'c', 
                            .packages = "ranger") %dopar% {
                              
                              #fit model
                              m.i <- ranger::ranger(data = Train,
                                                    dependent.variable.name = "Mental_Health",
                                                    num.trees = num.trees,
                                                    mtry = mtry,
                                                    min.node.size = min.node.size)
                              
                              #returning prediction error as percentage
                              return(m.i$prediction.error * 100)
                              
                            }

#==============================================================================#
# Best mtry
#==============================================================================#

myGrid$prediction.error <- prediction.error

best.hyperparameters <- myGrid %>% 
  dplyr::arrange(prediction.error) %>% 
  dplyr::slice(1)

#==============================================================================#
# Calculating the best min.node.size
#==============================================================================#

myGrid <- expand.grid(num.trees = best.hyperparameters$num.trees,
                      mtry = best.hyperparameters$mtry,
                      min.node.size = seq(13, 19, 2)) #Can come back in and tighten this search window for the second round

set.seed(24)
#fitting each rf model with different hyperparameters
prediction.error <- foreach(num.trees = myGrid$num.trees,
                            mtry = myGrid$mtry,
                            min.node.size = myGrid$min.node.size,
                            .combine = 'c', 
                            .packages = "ranger") %dopar% {
                              
                              #fit model
                              m.i <- ranger::ranger(data = Train,
                                                    dependent.variable.name = "Mental_Health",
                                                    num.trees = num.trees,
                                                    mtry = mtry,
                                                    min.node.size = min.node.size)
                              
                              #returning prediction error as percentage
                              return(m.i$prediction.error * 100)
                              
                            }

#==============================================================================#
# Best min.node.size
#==============================================================================#

myGrid$prediction.error <- prediction.error

best.hyperparameters <- myGrid %>% 
  dplyr::arrange(prediction.error) %>% 
  dplyr::slice(1)

#==============================================================================#
# Fit the best model
#==============================================================================#

#fit model
set.seed(24)
m.i <- ranger::ranger(data = Train,
                      dependent.variable.name = "Mental_Health",
                      importance = "permutation",
                      mtry = best.hyperparameters$mtry,
                      num.trees = best.hyperparameters$num.trees,
                      min.node.size = best.hyperparameters$min.node.size)

parallel::stopCluster(cl = my.cluster)

#==============================================================================#
# How does the model perform on the test data
#==============================================================================#

probabilities = predict(m.i, data = Test)$predictions
rmse = caret::RMSE(Test$Mental_Health, probabilities)

#==============================================================================#
# Plot Actual vs Predicted
#==============================================================================#

probs <- data.frame(probabilities)

model <- data.frame(prediction = probs$probabilities,
                    Outcome = Test$Mental_Health)

ggplot(model, aes(x = prediction, y = Outcome)) + 
  geom_point() +
  stat_smooth(method = "lm",
              se = TRUE) +
  stat_cor(label.y = 6, 
           aes(label = paste(..rr.label.., sep = "~`,`~"))) +
  stat_regline_equation(label.y = 6.5)

#==============================================================================#
# Variable importance
#==============================================================================#

importance_to_df <- function(model){
  x <- as.data.frame(model$variable.importance)
  x$variable <- rownames(x)
  colnames(x)[1] <- "importance"
  rownames(x) <- NULL
  return(x)
}

importance.scores <- foreach(i = 1:1000, 
                             .combine = 'rbind', 
                             .packages = "ranger") %dopar% {
                               
                               #fit model
                               m.i <- ranger::ranger(data = Train,
                                                     dependent.variable.name = "Class",
                                                     importance = "permutation",
                                                     mtry = best.hyperparameters$mtry,
                                                     num.trees = best.hyperparameters$num.trees,
                                                     min.node.size = best.hyperparameters$min.node.size)
                               
                               #format importance
                               m.importance.i <- importance_to_df(model = m.i)
                               
                               #returning output
                               return(m.importance.i)
                               
                             }

parallel::stopCluster(cl = my.cluster)

#==============================================================================#
# Plot the variable importance
#==============================================================================#

ggplot2::ggplot(data = importance.scores) + 
  ggplot2::aes(y = reorder(variable, importance), 
               x = importance) +
  ggplot2::geom_boxplot() + 
  ggplot2::ylab("")
