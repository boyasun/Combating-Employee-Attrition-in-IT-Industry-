install.packages("Amelia")
install.packages("corrplot")
install.packages("splitstackshape")
install.packages("rsample")

library(dplyr)
library(Amelia)
library(tidyvaerse)
library(ggplot2)
library(Hmisc)
library(corrplot)
library(psych)
library(caret)
library(plyr)
library(repr)
library(randomForest)
require(caTools)
library(splitstackshape)
library(rsample)              #for stratify sampling
library(class)                #for classification


setwd("C:/Users/ibm-hr-analytics-attrition-dataset")
attrition = read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv" , na.strings = c(""))


# Data pre-processing

#checking for NA values 
sapply(attrition,function(x) class(x))

#checking for missing values
missmap(attrition, main = "Missing values vs observed")

#converting numeric categorical variables to factors
attrition$Education <-   as.factor(attrition$Education)
attrition$EnvironmentSatisfaction <-   as.factor(attrition$EnvironmentSatisfaction)
attrition$JobLevel <-   as.factor(attrition$JobLevel)
attrition$JobInvolvement <-   as.factor(attrition$JobInvolvement)
attrition$JobSatisfaction <-   as.factor(attrition$JobSatisfaction)
attrition$PerformanceRating <-   as.factor(attrition$PerformanceRating)
attrition$RelationshipSatisfaction <-   as.factor(attrition$RelationshipSatisfaction)
attrition$StockOptionLevel <-   as.factor(attrition$StockOptionLevel)
attrition$WorkLifeBalance <-   as.factor(attrition$WorkLifeBalance)

# dropping irrelavant columns- over18, EmployeeCount, EmployeeNumber

data <- subset(attrition,select=c(1:8,11:26,28:35))

#exploring relationships within data

#correlation
data_numeric <- select_if(data, is.numeric)
cordata <- cor(data_numeric)
corrplot(cordata)

#distribution
multi.hist(data_numeric, bcol="orange",dcol = "black")
# here we see we need to transform few variables few variables before fitting them into logistic regression model

boxplot(data_numeric$MonthlyIncome)

boxplot.matrix(data_numeric,use.cols = TRUE)

?boxplot.matrix()
#histBy(data,"MonthlyIncome","OverTime")

data_cat <- select_if(data,is.factor)
count(data_cat,Attrition)
#checking y variable
data_cat%>%group_by(Attrition)%>%summarise(count = n())

# distribution of attribution rate across different factors
ggplot(data = data,aes(fill = Attrition, x = JobSatisfaction )) + geom_bar(position = "fill")

ggplot(data = data,aes(fill = Attrition, x = Gender )) + geom_bar(position = "fill")

ggplot(data = data,aes(fill = Attrition, x = MaritalStatus )) + geom_bar(position = "fill")
#attrtionrate doubles if they are single
ggplot(data = data,aes(fill = Attrition, x = JobInvolvement )) + geom_bar(position = "fill")
#clear correlation
ggplot(data = data,aes(fill = Attrition, x = OverTime )) + geom_bar(position = "fill")

#variation and central tendancy of attrition measure
ggplot(data  = data , aes(x = Attrition, y = Age)) + geom_boxplot()
ggplot(data  = data , aes(x = Attrition, y = log(MonthlyIncome))) + geom_boxplot()

#### Model Development
set.seed(100)

data1 = subset(data,select = c(1:3,5:19,21:32))
data1$Attrition <- ifelse(data1$Attrition == "Yes", 1, 0)


#sample data creation

sample_index <- initial_split(data1,prop = 3/4,strata = Attrition)
train <- training(sample_index)
test <- testing(sample_index)


#function to calculate R2

# Compute R^2 from true and predicted values
eval_results <- function(true, predicted, df) {
  SSE <- sum((predicted - true)^2)
  SST <- sum((true - mean(true))^2)
  R_square <- 1 - SSE / SST
  RMSE = sqrt(SSE/nrow(df))
  
  # Model performance metrics
  data.frame(
    RMSE = RMSE,
    Rsquare = R_square
  )
  
}

# Elastic net regularization

train_cont <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 5,
                           search = "random",
                           verboseIter = TRUE)

# Train the model
elastic_reg <- train(Attrition ~ .,
                     data = data1,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneLength = 10,
                     trControl = train_cont)

?trainControl
#Fitting alpha = 0.177, lambda = 0.00212 on full training set
# Best tuning parameter
elastic_reg$coefnames

colnames(test)
y_test = test$Attrition
x_test = test[,-2]
# Make predictions on training set
predictions_train <- predict(elastic_reg, x_test)

eval_results(y_test, predictions_train, test) 

library(readr)
# Make predictions on test set
predictions_test <- predict(elastic_reg, x_test)
eval_results(y_test, predictions_test, test)


# KNN Classification
nor <-function(x) { (x -min(x))/(max(x)-min(x))}

data2 <- subset(data,select = c(1:3,5:10,11:16,18,20:27,32))
data2_knn <- as.data.frame(lapply(data_numeric, nor))
data2_final <- subset(cbind(data2_knn,data_cat),select = c(1,3,5,7:11,15:26,28:32))

#sampling

sample_knn = initial_split(data2_final,prop  = 0.75,strata = Attrition)
train <- training(sample_knn)
test <- testing(sample_knn)

x_train <- train
y_train <- train[,9]
x_test <- test[,-9]
y_test <- test[,9]

pr <- knn(x_train,x_test,cl= y_train,k=20)

knn

#Random Forest 
#https://www.rdocumentation.org/packages/rfUtilities/versions/2.1-5/topics/rf.crossValidation

sample = sample.split(data1$num, SplitRatimsjjo = .75)
train = subset(data1, sample == TRUE)
test  = subset(data1, sample == FALSE)
dim(train)
dim(test)

rf <- randomForest(
  Attrition ~ .,
  data= train,ntree = 500,mtry = 4)
importance(rf)
varImpPlot(rf)

rf

pred = predict(rf, newdata=test[-9])
cm = table(test[,9], pred)

#Confusion matrix: No. of trees 500,
#        0  1  class.error
#0   1225  8  0.00648824
#1    193 44  0.81434599

        
#TN = 1225, TP = 44, FP = 8 , TP = 44

     
