library(dplyr)
library(data.table)
library(caret)
library(mgcv)

Data <- read.csv(url("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"),
                     header = FALSE)

#==============================================================================#
#
#==============================================================================#

# Fix the names of the data if needed
Data <- Data %>%
  rename(ID = V1,
         thick = V2,
         size = V3,
         shape = V4,
         marg = V5,
         single = V6,
         bare = V7,
         bland = V8,
         normal = V9,
         mitoses = V10,
         Class = V11)

# Remove the NA rows
Data <- Data[complete.cases(Data),]

# Turn the outcome varialbe into a factor 
Data$location <- as.factor(Data$location)

Data = subset(Data, select = -c(Name) )
levels(Data$location) <- c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

#Data$Class <- fifelse(Data$Class == 2, "B", "M")

# Split the data for training and testing
set.seed(24)
trainIndex <- createDataPartition(Data$location, 
                                  p = .8, #(80/20 split of the data)
                                  list = FALSE, 
                                  times = 1)

Train <- as.data.frame(Data[ trainIndex,])
Test  <- as.data.frame(Data[-trainIndex,])

#==============================================================================#
# GLM 
#==============================================================================#

fit.control <- trainControl(method = "repeatedcv", 
                            number = 5, 
                            repeats = 10,
                            #summaryFunction = twoClassSummary, 
                            classProbs = TRUE)

set.seed(24)
GLM_model <- train(V37 ~ .,
                   data = Train,
                   method = "glm",
                   trControl = fit.control,
                   family = binomial(link = "logit"))

set.seed(24)
GAM_model <- train(V37 ~ .,
                   data = Train,
                   method = "gam",
                   trControl = fit.control,
                   preProcess = c('center', 'scale'),
                   metric = "Accuracy",
                   family = "binomial")

rs <- resamples(list(GLM = GLM_model, GAM = GAM_model))

#==============================================================================#
# GLM Model evaluation
#==============================================================================#

# Prediction
GLM.pred <- predict(GLM_model, Test)

# Confusion Matrix
confusionMatrix(as.factor(GLM.pred), as.factor(Test$Class),
                mode = "everything")

#==============================================================================#
#  GAM Model evaluation
#==============================================================================#

# Prediction
GAM.pred <- predict(GAM_model, Test)

# Confusion Matrix
confusionMatrix(as.factor(GAM.pred), as.factor(Test$Class))

