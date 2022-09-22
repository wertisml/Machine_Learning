library(dplyr)
library(data.table)
library(caret)
library(earth)
library(vip)       
library(pdp)
library(ggplot2)
library(broom)

Data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"),
                 header = F)

#==============================================================================#
# Data preparation
#==============================================================================#

# Fix the names of the data if needed
Data <- Data %>%
  rename(Class = V1,
         alcohol = V2,
         malic = V3,
         ash = V4,
         alcalinity = V5,
         magn = V6,
         total = V7,
         flavor = V8,
         nonfalvor = V9,
         pro = V10,
         color = V11,
         Hue = V12,
         OD280 = V13,
         Pro = V14,
         location = V15)

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Turn the outcome varialbe into a factor 
Data$location <- as.factor(Data$location)

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$location, 
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
# MARS
#==============================================================================#

# Part 1
trControl <- trainControl(method = "cv",
                          number = 10)

myGrid <- expand.grid(degree = 1:3, 
                      nprune = seq(2, 100, length.out = 10) 
                      %>% floor())

set.seed(24)
MARS_model <- train(x = Train[,-15],
                    y = Train[,15],
                    method = "earth",
                    #metric = "RMSE",
                    trControl = trControl,
                    tuneGrid = myGrid)

MARS_model$bestTune

ggplot(MARS_model)

# Part 2

# Tune the grid based on the ggplot results
tuned_grid <- expand.grid(degree = 1:3, 
                      nprune = c(10,11,12,13,14,15,16,17,18,19,20,21,22,23) 
                      %>% floor())

set.seed(24)
MARS_model2 <- train(x = Train[,-10],
                    y = Train[,10],
                    method = "earth",
                    #metric = "RMSE",
                    trControl = trControl,
                    tuneGrid = tuned_grid)

MARS_model2$bestTune

# You might have to rerun tuned_grid depending on the results of the best nprune
ggplot(MARS_model2)

#==============================================================================#
#  MARS Model evaluation
#==============================================================================#

# Prediction
MARS_model2 <- predict(MARS_model2, Test)

# Confusion Matrix
confusionMatrix(as.factor(MARS_model2), as.factor(Test$Class))

#==============================================================================#
# variable importance plots
#==============================================================================#

p1 <- vip(MARS_model2, num_features = 13, geom = "point", value = "gcv") + ggtitle("GCV")
p2 <- vip(MARS_model2, num_features = 13, geom = "point", value = "rss") + ggtitle("RSS")

gridExtra::grid.arrange(p1, p2, ncol = 2)


