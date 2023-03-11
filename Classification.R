


#Classification And REgression Training
library(caret)
library(tidyverse)
library(AppliedPredictiveModeling)
library(pls) 
library(elasticnet)
library(broom) 
library(glmnet)
library(MASS)
library(ISLR)
library(PerformanceAnalytics)
library(Matrix) 
library(kernlab) 
library(e1071) #svm 
library(rpart) #cart 

library(dslabs)
library(rpart.plot) #rpart 
library(partykit) 
library(ipred) #bagging 
library(randomForest)
library(gbm)
library(nnet)
library(neuralnet)
library(GGally)
library(NeuralNetTools) 
library(FNN)
library("readxl")
library(SHAPforxgboost)
library(xgboost)
library(data.table)
library(ggplot2)
library(hrbrthemes)
library(ggraph)
library(igraph)
library(tidyverse)
library(viridis)
library(corrgram)
library(ggplot2)
library(viridis)
library(hrbrthemes)
library(ggridges)
library(h2o)

## v

df <- read_xlsx('C:/Users/berka/Desktop/masaüstü_2023/freelance/ysa/data25.xlsx')

df$Passing_Covid

df <- df %>%
  mutate(Passing_Covid = ifelse(Passing_Covid == "1","Covid","No_Covid"))



set.seed(123)


train_indeks <- createDataPartition(df$Passing_Covid, p = 0.7, list = FALSE, times = 1)

train <- df[train_indeks,]
test <- df[-train_indeks,]

train_x <- train %>% dplyr::select(-Passing_Covid)
train_y <- train$Passing_Covid

test_x <- test %>% dplyr::select(-Passing_Covid)
test_y <- test$Passing_Covid

training <- data.frame(train_x, Passing_Covid = train_y)

knn_train <- train
knn_test <- test

train_y<-as.factor(train_y)


knn_train_x <- train %>% dplyr::select(-Passing_Covid)
knn_train_y <- train$Passing_Covid

knn_test_x <- test %>% dplyr::select(-Passing_Covid)
knn_test_y <- test$Passing_Covid

levels(train$Passing_Covid) <- make.names(levels(factor(train$Passing_Covid)))
knn_train_y <- train$Passing_Covid


library(caret)
set.seed(123)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

knn_grid <- data.frame(k = 1:30)

knn_tune <- caret::train(train_x, train_y,
                         method = "knn",
                         metric = "ROC",
                         preProc = c("center", "scale"),
                         trControl = ctrl,
                         tuneGrid = knn_grid)



plot(knn_tune, main = "Best Parameter Value for KNN", xlab= "Neighbors")


predtrain <- predict(knn_tune, train_x)
caret::confusionMatrix(factor(predtrain), factor(train_y), positive = "No_Covid")

predtest <- predict(knn_tune, test_x)
caret::confusionMatrix(factor(predtest), factor(test_y), positive = "No_Covid")




library(pROC)
library(ROCR) #roc icin
roc(knn_tune$pred$obs,
    knn_tune$pred$No_Covid,
    levels = rev(levels(knn_tune$pred$obs)),
    plot = TRUE, print.auc = TRUE)



# Destek Vektör Regresyonu (SVR)

svm_train_x <- train %>% dplyr::select(-Passing_Covid)
svm_train_y <- train$Passing_Covid

svm_test_x <- test %>% dplyr::select(-Passing_Covid)
svm_test_y <- test$Passing_Covid

levels(train$Passing_Covid) <- make.names(levels(factor(train$Passing_Covid)))
svm_train_y <- train$Passing_Covid
## Model



library(e1071)
library(kernlab)
library(purrr)
library(ggplot2)
library(PerformanceAnalytics)





set.seed(123)


ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

svm_grid <- expand.grid(sigma = c(0.01, 0.015, 0.2,0.25,0.3),
                        C = c(0,0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25,1.5,2,2.5,3,3.5,4))

svm_tune <- caret::train(train_x, train_y,
                         method = "svmRadial",
                         metric = "ROC",
                         tuneGrid = svm_grid,
                         trControl = ctrl)


plot(svm_tune, main = "Best Parameter Value for SVM", xlab= "Cost")


predtrain <- predict(svm_tune, svm_train_x)
caret::confusionMatrix(factor(predtrain), factor(svm_train_y), positive = "No_Covid")

predtest <- predict(svm_tune, svm_test_x)
caret::confusionMatrix(factor(predtest), factor(svm_test_y), positive = "No_Covid")



svm_tune$finalModel




# YSA

## Data Set and Preparation

ysa_train_x <- train %>% dplyr::select(-Passing_Covid)
ysa_train_y <- train$Passing_Covid

ysa_test_x <- test %>% dplyr::select(-Passing_Covid)
ysa_test_y <- test$Passing_Covid

levels(train$Passing_Covid) <- make.names(levels(factor(train$Passing_Covid)))
ysa_train_y <- train$Passing_Covid

## Model Tuning



library(RSNNS)

set.seed(123)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

ysa_grid <- expand.grid(
  decay = c(0.001,0.01,0.05, 0.1, 0.2, 0.4),
  size =  (1:10), bag = c(T,F))

ysa_tune <- caret::train(train_x, train_y,
                         method = "avNNet",
                         trControl = ctrl,
                         tuneGrid = ysa_grid,
                         preProc = c("center", "scale"),
                         linout = TRUE, maxit = 100)

plot(ysa_tune, main = "Best Parameter Value for ANN", xlab= "Hidden Units")


predtrain <- predict(ysa_tune, ysa_train_x)
caret::confusionMatrix(factor(predtrain), factor(ysa_train_y), positive = "No_Covid")

predtest <- predict(ysa_tune, ysa_test_x)
caret::confusionMatrix(factor(predtest), factor(ysa_test_y), positive = "No_Covid")

# Random Forests Regresyon




set.seed(123)


rf_train_x <- train %>% dplyr::select(-Passing_Covid)
rf_train_y <- train$Passing_Covid

rf_test_x <- test %>% dplyr::select(-Passing_Covid)
rf_test_y <- test$Passing_Covid

levels(train$Passing_Covid) <- make.names(levels(factor(train$Passing_Covid)))
rf_train_y <- train$Passing_Covid

ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

tune_grid <- expand.grid(mtry = c(1,2,3,4,5,8,10,12,15,17,20))
rf_tune <- caret::train(train_x, train_y,
                        method = "rf",
                        tuneGrid = tune_grid,
                        trControl = ctrl
                        
)

rf_tune
plot(rf_tune, main = "Best Parameter Value for RF", xlab= "Randomly Selected Predictors")

rf_tune$results %>% filter(mtry == as.numeric(rf_tune$bestTune))

predtrain <- predict(rf_tune, rf_train_x)
caret::confusionMatrix(factor(predtrain), factor(rf_train_y), positive = "No_Covid")

predtest <- predict(rf_tune, rf_test_x)
caret::confusionMatrix(factor(predtest), factor(rf_test_y), positive = "No_Covid")



#XGBoost
## Model 


library(xgboost)
set.seed(123)
ctrl <- trainControl(method = "cv",
                     number = 10,
                     summaryFunction = twoClassSummary, 
                     classProbs = TRUE)

xgb_grid <- expand.grid(eta = c(0.03,0.05,0.1,0.5), 
                        nrounds = c(100),  
                        max_depth = 1:7,  
                        min_child_weight = c(1,2),  
                        colsample_bytree = c(0.3,0.5), 
                        gamma = 0, 
                        subsample = 1)




xgb_tune <- caret::train(x = as.matrix(train_x),
                         y = as.matrix(train_y),
                         method = "xgbTree",
                         tuneGrid = xgb_grid,
                         trControl = ctrl,
                         metric = "ROC")
xgb_tune$bestTune
plot(xgb_tune)
plot(xgb_tune, main = "Best Parameter Value for XGBoost")

pred <- predict(xgb_tune, test_x)

pred <- factor(pred)

caret::confusionMatrix(pred, factor(test_y), positive = "No_Covid")



# Regression
## Model Kurma


## Model Tuning


set.seed(123)
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

eGrid <- expand.grid(.alpha = (0:10) * 0.1, 
                     .lambda = (0:10) * 0.1)


reg_tune <- caret::train(data.matrix(train_x), train_y,
                         method = "glmnet",
                         tuneGrid = eGrid,
                         trControl = ctrl,
                         metric = "ROC")


reg_tune$bestTune



plot(reg_tune, main = "Best Parameter Value for Regression")

roc(reg_tune$pred$obs,
    reg_tune$pred$No_Covid,
    levels = rev(levels(reg_tune$pred$obs)),
    plot = TRUE, print.auc = TRUE)



pROC_obj <- roc(as.numeric(factor(test_y)),as.numeric(pred),
                smoothed = TRUE,
                # arguments for ci
                ci=TRUE, ci.alpha=0.9, stratified=FALSE,
                # arguments for plot
                plot=TRUE, auc.polygon=TRUE, max.auc.polygon=TRUE, grid=TRUE,
                print.auc=TRUE, show.thres=TRUE)


sens.ci <- ci.se(pROC_obj)
plot(sens.ci, type="shape", col="lightblue")
## Warning in plot.ci.se(sens.ci, type = "shape", col = "lightblue"): Low
## definition shape.
plot(sens.ci, type="bars")

library(PRROC)

PRROC_obj <- roc.curve(scores.class0 = as.numeric(pred), weights.class0 =as.numeric(factor(test_y)),
                       curve=TRUE)
plot(PRROC_obj)

pred <- predict(reg_tune, test_x)

pred <- factor(pred)

caret::confusionMatrix(pred, factor(test_y), positive = "No_Covid")



# Cart
## Model Kurma



cart_train_x <- train %>% dplyr::select(-Passing_Covid)
cart_train_y <- train$Passing_Covid

cart_test_x <- test %>% dplyr::select(-Passing_Covid)
cart_test_y <- test$Passing_Covid

levels(train$Passing_Covid) <- make.names(levels(factor(train$Passing_Covid)))
cart_train_y <- train$Passing_Covid


set.seed(123)

ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     summaryFunction = twoClassSummary,
                     classProbs = TRUE,
                     savePredictions = TRUE)

tune_grid = expand.grid(cp=c(0,0.001,0.002,0.003,0.004,0.005,0.007,0.01,0.012,0.015,0.017))


# Use the train() function to create the model
cart_tune <- caret::train(Passing_Covid ~.,
                          data=train,                 # Data set
                          method="rpart",                     # Model type(decision tree)
                          trControl= ctrl,           # Model control options
                          tuneGrid = tune_grid,               # Required model parameters
                          maxdepth = 5,                       # Additional parameters***
                          minbucket=5,
                          metric = "ROC")  
cart_tune

plot(cart_tune, main = "Best Parameter Value for CART")


plot(predict(cart_tune), train$Passing_Covid, xlab = "Tahmin",ylab = "Gercek")
abline(0,1)



defaultSummary(data.frame(obs = train$Passing_Covid,
                          pred = predict(cart_tune)))



# Sonuc

KNN <- caret::confusionMatrix(predict(knn_tune, test_x), factor(test_y), positive = "No_Covid")


knn_tune$finalModel



SVM<-caret::confusionMatrix(predict(svm_tune, test_x), factor(test_y), positive = "No_Covid")

SVM
svm_tune$finalModel



ANN<-caret::confusionMatrix(predict(ysa_tune, test_x), factor(test_y), positive = "No_Covid")

ANN
ysa_tune$finalModel
ysa_tune$bestTune


RF<-caret::confusionMatrix(predict(rf_tune, test_x), factor(test_y), positive = "No_Covid")

RF
rf_tune$finalModel



XGBoost<-caret::confusionMatrix(predict(xgb_tune, test_x), factor(test_y), positive = "No_Covid")

XGBoost
xgb_tune$bestTune


CART<-caret::confusionMatrix(predict(cart_tune, test_x), factor(test_y), positive = "No_Covid")

CART
cart_tune$finalModel
cart_tune$bestTune


REG<-caret::confusionMatrix(predict(reg_tune, test_x), factor(test_y), positive = "No_Covid")

REG
reg_tune$bestTune

KNN
SVM
ANN
RF
XGBoost
CART
REG





toplam<- rbind((KNN$overall["Accuracy"]),CART$overall["Accuracy"],SVM$overall["Accuracy"]
               ,ANN$overall["Accuracy"],RF$overall["Accuracy"],XGBoost$overall["Accuracy"],
               REG$overall["Accuracy"])
Output <- data.frame( Method = c("KNN","CART","SVM","ANN","RF","XGBoost","REG"))
Output2 <- data.frame( Comparison = c("Accuracy"))
toplam<- cbind(Output,toplam,Output2)
toplam<-as.data.frame(toplam)

toplam<-rename(toplam,Values=Accuracy)


toplam2<- rbind((KNN$byClass["F1"]),CART$byClass["F1"],SVM$byClass["F1"],ANN$byClass["F1"],
                RF$byClass["F1"],XGBoost$byClass["F1"],REG$byClass["F1"])
Output3 <- data.frame( Method = c("KNN","CART","SVM","ANN","RF","XGBoost","REG"))
Output4 <- data.frame( Comparison = c("F1"))
toplam2<- cbind(Output3,toplam2,Output4)
toplam2<-as.data.frame(toplam2)

toplam2<-rename(toplam2,Values=F1)

toplam3<- rbind((KNN$byClass["Sensitivity"]),CART$byClass["Sensitivity"],SVM$byClass["Sensitivity"],
                ANN$byClass["Sensitivity"],RF$byClass["Sensitivity"],XGBoost$byClass["Sensitivity"],
                REG$byClass["Sensitivity"])
Output5 <- data.frame( Method = c("KNN","CART","SVM","ANN","RF","XGBoost","REG"))
Output6 <- data.frame( Comparison = c("Sensitivity "))
toplam3<- cbind(Output5,toplam3,Output6)
toplam3<-as.data.frame(toplam3)

toplam3<-rename(toplam3,Values=Sensitivity )

toplam <- rbind(toplam, toplam2,toplam3)

ggplot(toplam,                 
       aes(x = Method,
           y = Values,
           fill = Comparison)) +
  geom_bar(stat = "identity",
           position = "dodge")+labs(fill = "Metrics")


ggplot(toplam, aes(x = Method, y = Values, 
                   color = Comparison, group = Comparison)) + 
  geom_line() + geom_point()+labs(fill = "Metrics")



test_pred_grid<-predict(rf_tune, test_x)
tahmin<-cbind(test_pred_grid,
              test_y)
tahmin<-as.data.frame(tahmin)
colnames(tahmin) <- c('Prediction','Observation')
tahmin <- tahmin %>% gather(tahmin1, deger, Prediction:Observation)




