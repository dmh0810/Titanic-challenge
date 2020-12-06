# Titanic-challenge
This is my very first project and thus the very first attempt to implement several Machine Learning algorithms. During my internship at the StuExplorative data analysis, data visualization , logistic regression, tree-based models (decision trees, bagged trees, random forest, boosted trees), k-nearest neighbors, naive bayes and support vector machine in R

### Daten einlesen
titanic <- read.csv(file.choose(), sep = ";", na = c("", "NA"))

install.packages("dplyr")
install.packages("ggplot2")
install.packages("stringr")

library(dplyr)
library(ggplot2)
library(stringr)

#View data

titanic$ID <- 1:nrow(titanic)

titanic <- titanic[1:1309,] 

str(titanic)

colSums(is.na(titanic))

# Data cleaning
  # sibsp: Anzahl der Geschwister, die zusammen am Bord sind
  # parch: Anzahl der Elternteile und Familienmitglieder 

titanic$age <- str_replace_all(as.character(titanic$age), ",", ".")
titanic$fare <- str_replace_all(as.character(titanic$fare), ",", ".")

  # Missings behandle: Age mit median, Ticketpreis mit Durschnitt 
titanic$age[is.na(titanic$age)] <- median(as.numeric(titanic$age), na.rm = TRUE)

titanic$fare[is.na(titanic$fare)] <- mean(as.numeric(titanic$fare), na.rm = TRUE)

titanic$pclass <- as.factor(titanic$pclass)

titanic$age <- as.numeric(titanic$age)
titanic$fare <- as.numeric(titanic$fare)

titanic$survived <- as.factor(titanic$survived)

# Visualisationen: 
titanic%>% group_by(survived) %>% summarise(n = n())

  #Pclass Plot
ggplot(titanic, aes(x=pclass, fill = survived)) + geom_histogram(stat = "count")
ggplot(titanic, aes(x= pclass, y = fare, fill = survived)) + geom_boxplot()  

  #Geschlecht plot
ggplot(titanic, aes(x=sex, fill = survived)) + geom_histogram(stat = "count")

  # Alter und Geschlecht 
ggplot(titanic, aes(x= age, fill = survived)) +
  geom_density(alpha = 0.5)+
  facet_grid(~sex)
ggplot(titanic, aes(x= sex, y = age, fill = survived)) + geom_boxplot()

  #Ticketpreis 
ggplot(titanic, aes(x = fare, fill = survived)) + geom_density(alpha = 0.5)

  #Embarked (wo eingestiegen)
ggplot(titanic, aes(x = embarked, fill = survived)) + geom_histogram(stat = "count")

  #Family size 
ggplot(titanic, aes(x=sibsp, fill = survived)) +geom_histogram(stat = "count")

ggplot(titanic, aes(x=parch, fill = survived)) + geom_histogram(stat = "count")

# Test/train spilt (418/891)
set.seed(1)

n_obs <- nrow(titanic)

permuted_row <- sample(n_obs)

titanic_shuffled <- titanic[permuted_row,]

titanic_train <- titanic_shuffled[1:891,]

titanic_test <- titanic_shuffled[892:nrow(titanic_shuffled),]

#Zufallsstichprobe: Prüfung der Representative

titanic_train %>% group_by(survived) %>% summarise(n = n())

titanic_test %>% group_by(survived)%>% summarise(n = n())

#Umcodierung der predictors 
titanic$sex <- if_else(titanic$sex == "male", 1, 0)

titanic_train$sex <- if_else(titanic_train$sex == "male", 1, 0)

titanic_test$sex <- if_else(titanic_test$sex == "male", 1, 0)

titanic2 <- titanic %>% mutate(survived = if_else(titanic$survived == 1, "yes", "no"))
titanic2$survived <- as.factor(titanic2$survived)


# Logit-Model 
install.packages("caret")
install.packages("caTools")

library(caTools)
library(Metrics)
library(caret)

set.seed(1)

model_logit <- glm(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, family = "binomial", titanic_train)

summary(model_logit)

# Interpretation der Koeffizienten: 
coefficients(model_logit)

  # Wenn ein Passenger männlich ist, sinkt die Überlebenswahrscheinlichkeit um: 
paste0("Wenn ein Passenger männlich ist, sinkt die Überlebenswahrscheinlichkeit auf: ", exp(1)^-2.5992389104)
  
  # Wenn ein Passenger von 1. Klasse auf 2.Klasse wechselt, sinkt die Überlebenswahrscheinlichkeit um: 
paste0(" Wenn ein Passenger von 1. Klasse auf 2.Klasse wechselt, sinkt die Überlebenswahrscheinlichkeit auf:",exp(1)^-0.9311152453) 

  #Wenn ein Passenger ein Jahr älter wäre, sinkt die Überlebenswahrscheinlichkeit um: 
paste0("Wenn ein Passenger ein Jahr älter wäre, sinkt die Überlebenswahrscheinlichkeit auf: ",exp(1)^-0.0442696043)

# Prognosen treffen 
logit_pred <- predict(model_logit, newdata = titanic_test, type = "response")

# Predictions using threshold = 0.4 für accuracy measure
p <- if_else(logit_pred > 0.4, 1, 0)

p_class <- factor(p, levels = levels(as.factor(titanic_test[["survived"]])))

  # Confusion Matrix 
confusionMatrix(p_class, as.factor(titanic_test[["survived"]]))

# Pseudo R2
install.packages("desc")
library(descr)

LogRegR2(model_logit)

# Cross validation
ctrl <- trainControl(method = "cv",     
                     number = 5,                            
                     classProbs = TRUE,                  
                     summaryFunction = twoClassSummary)  

logit_model_cv <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked,
                        data = titanic2, 
                        method = "glm",
                        metric = "ROC",
                        trControl = ctrl, 
                        na.action = na.omit)

logit_model_cv$results[,"ROC"]

# Bäume
#a. Decision tree: 
install.packages("rpart")
install.packages("rpart.plot")

library(rpart)
library(rpart.plot)
set.seed(1)

model_class_tree <- rpart(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, titanic_train, method = "class")

# Prune the model (to optimized cp value)
 
    #Modify cp
plotcp(model_class_tree)
opt_index <- which.min(model_class_tree$cptable[, "xerror"])
cp_opt <- model_class_tree$cptable[opt_index, "CP"]  
sprintf("optimale cp: %.3f", cp_opt)
class_tree_model_opt <- prune(tree = model_class_tree,  cp = cp_opt)  

  # Plot the optimized model
rpart.plot(x = class_tree_model_opt, yesno = 2, type = 0, extra = 0)

# Optimize decision tree model

model_class_tree_mod <- rpart(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, titanic_train, method = "class", cp = cp_opt)

pred_class_tree_mod <- predict(model_class_tree_mod, titanic_test, type = "class")

pred_class_tree_prob <- predict(model_class_tree_mod, titanic_test, type = "prob")

p_class_tree_mod <- factor(pred_class_tree_mod, levels = levels(as.factor(titanic_test[["survived"]])))

#Confusion Matrix decision tree
confusionMatrix(p_class_tree_mod, as.factor(titanic_test[["survived"]]))

#b. Random Forest
install.packages("randomForest")

library(randomForest)

set.seed(1)

titanic_train$survived <- as.factor(titanic_train$survived)
titanic_test$survived <- as.factor(titanic_test$survived)

model_rf_raw <- randomForest(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, data = titanic_train ,na.action = na.omit)

# OOB error
err <- model_rf_raw$err.rate

oob_err <- err[nrow(err), "OOB"]
    # How many trees are optimal?
plot(model_rf_raw)
legend(x="right", legend = colnames(err), fill = 1:ncol(err)) #300 trees optimal 

#Tuning Random forest
train_rf_tune <- titanic_train %>% select(pclass,sex,age,sibsp,parch,fare,embarked, survived) %>% filter(!is.na(embarked)) 

res <- tuneRF(x = subset(train_rf_tune, select = -survived), 
              y = train_rf_tune$survived,
              ntreeTry = 300)         

    # Find the mtry that minimizes OOB error (mtry: Anzahl der prognostizierten Variablen pro Split)
    #mtry: in dem Paket randomforest gibt es eine built-in Funktion zum Tuning von mtry 
mtry_opt <- res[,"mtry"][which.min(res[,"OOBError"])]
sprintf("optimale Anzahl der Variablen pro Split: %.3f", mtry_opt)
    
    # Modify rf model
rf_model_new <- randomForest(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, data = titanic_train ,na.action = na.omit, mtry = mtry_opt)

summary(rf_model_new)

pred_rf_prob <- predict(rf_model_new, titanic_test, type = "prob")
pred_rf <- predict(rf_model_new, titanic_test, type = "class")

#Confusion Matrix rf
confmax_rf <- confusionMatrix(pred_rf, titanic_test$survived)

#Test set error
paste0("Test Accuracy: ", confmax_rf$overall[1])

paste0("OOB Accuracy: ", 1 - oob_err)

pred_rf_num <- predict(model_rf_raw, titanic_test, type = "prob")

#c. Bagged trees
install.packages("ipred")
library(ipred)

set.seed(1)

model_bagging <- bagging(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, data = titanic_train ,na.action = na.omit, coob = TRUE)

pred_bagging <- predict(model_bagging, titanic_test)

pred_bagging_prob <- predict(model_bagging, titanic_test, type = "prob")

#Confusion Matrix bagging
confusionMatrix(pred_bagging,titanic_test$survived)

#Cross validation 
ctrl <- trainControl(method = "cv",     
                     number = 5,                            
                     classProbs = TRUE,                  
                     summaryFunction = twoClassSummary)  
          # Track AUC (Area under the ROC curve)
model_bagging_cv <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked,
                        data = titanic2, 
                        method = "treebag",
                        metric = "ROC",
                        trControl = ctrl, 
                        na.action = na.omit)

model_bagging_cv$results[,"ROC"]

#d. GBM
install.packages("gbm")
library(gbm)

set.seed(1)

train3 <- titanic_train
train3$survived <- as.character(train3$survived)

test3 <- titanic_test
test3$survived <- as.character(test3$survived)

model_gbm <- gbm(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, distribution = "bernoulli", data = train3, n.trees=500)

summary(model_gbm)

# Early stopping GBM finden
  # based on OOB, um die optimale Anzahl von Bäumen zu finden
ntree_opt_oob <- gbm.perf(object = model_gbm, 
                          method = "OOB", 
                          oobag.curve = TRUE)
print(paste0("Optimal n.trees (OOB Estimate): ", ntree_opt_oob))                         

#Optimize GBM model
model_gbm_new <- gbm(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, distribution = "bernoulli", data = train3, n.trees= ntree_opt_oob)

pred_gbm <- predict(model_gbm, test3, n.trees= ntree_opt_oob, type = "response")

#Confusion matrix GBM
pred_gbm_class <- if_else(pred_gbm > 0.4, 1,0)

p_gbm <- factor(pred_gbm_class, levels = levels(as.factor(test3[["survived"]])))

confusionMatrix(p_gbm, as.factor(test3$survived))

#e. Compare ROCs and AUCs of different test models
logit <- auc(titanic_test$survived, logit_pred)

decision_tree <- auc(titanic_test$survived, pred_class_tree_mod)

bagged <- auc(titanic_test$survived, pred_bagging)

rf_auc <- auc(titanic_test$survived, pred_rf)

gbm <- auc(test3$survived, pred_gbm)

sprintf("Logistic regression AUC: %.3f", logit)
sprintf("Decision Tree Test AUC: %.3f", decision_tree)
sprintf("Bagged Trees Test AUC: %.3f", bagged)
sprintf("Random Forest Test AUC: %.3f", rf_auc)
sprintf("GBM Test AUC: %.3f", gbm)

# Plot all models
install.packages("ROCR")

library(ROCR)

preds_list <- list(logit_pred, pred_class_tree_prob[,1], pred_bagging_prob[,1], pred_rf_prob[,1], pred_gbm)
m <- length(preds_list)
actuals_list <- rep(list(titanic_test$survived),m)

# Plot the ROC curves
pred <- prediction(preds_list, actuals_list)

rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Logit", "Decision Tree", "Bagged Trees", "Random Forest", "GBM"),
       fill = 1:m)


# k-NN-Modell
# Daten vorbereiten: Dummy-Variablen hinzufügen und numerische Variablen normailisieren 

# k-NN Datensatz vorbereiten
knn_train <- titanic_train
knn_test <- titanic_test

knn_train$pclass <- as.numeric(knn_train$pclass)
knn_test$pclass <- as.numeric(knn_test$pclass)

# Variablen Pclass, Age, sibsp, parch und fare rescale: (0,1)
#Pclass
min_class_train <- min(knn_train$pclass) 
max_class_train <- max(knn_train$pclass)
knn_train$pclass <- (knn_train$pclass - min_class_train)/(max_class_train - min_class_train)

min_class_test <- min(knn_test$pclass)
max_class_test <- max(knn_test$pclass)
knn_test$pclass <- (knn_test$pclass - min_class_test)/ (max_class_test - min_class_test)

#Age
min_age_train <- min(knn_train$age)
max_age_train <- max(knn_train$age)
knn_train$age <- (knn_train$age - min_age_train)/(max_age_train - min_age_train)

min_age_test <- min(knn_test$age)
max_age_test <- max(knn_test$age)
knn_test$age <- (knn_test$age - min_age_test)/(max_age_test - min_age_test)

#sibsp
min_sibsp_train <- min(knn_train$sibsp)
max_sibsp_train <- max(knn_train$sibsp)
knn_train$sibsp <- (knn_train$sibsp - min_sibsp_train)/(max_sibsp_train - min_sibsp_train)

min_sibsp_test <- min(knn_test$sibsp)
max_sibsp_test <- max(knn_test$sibsp)
knn_test$sibsp <- (knn_test$sibsp - min_sibsp_test)/(max_sibsp_test - min_sibsp_test)

#parch
min_parch_train <- min(knn_train$parch)
max_parch_train <- max(knn_train$parch)
knn_train$parch <- (knn_train$parch - min_parch_train)/(max_parch_train - min_parch_train)

min_parch_test <- min(knn_test$parch)
max_parch_test <- max(knn_test$parch)
knn_test$parch <- (knn_test$parch - min_parch_test)/(max_parch_test - min_parch_test)

#Fare
min_fare_train <- min(knn_train$fare)
max_fare_train <- max(knn_train$fare)
knn_train$fare <- (knn_train$fare - min_fare_train)/(max_fare_train - min_fare_train)

min_fare_test <- min(knn_test$fare)
max_fare_test <- max(knn_test$fare)
knn_test$fare <- (knn_test$fare - min_fare_test)/(max_fare_test - min_fare_test)

#Dummy Variablen für Variable embarked 
knn_train$embarkedC <- if_else(titanic_train$embarked == "C", 1, 0)
knn_train$embarkedS <- if_else(titanic_train$embarked == "S", 1, 0)
knn_train$embarkedQ <- if_else(titanic_train$embarked == "Q", 1, 0)
knn_train$embarked <- NULL

knn_test$embarkedC <- if_else(titanic_test$embarked == "C", 1, 0)
knn_test$embarkedS <- if_else(titanic_test$embarked == "S", 1, 0)
knn_test$embarkedQ <- if_else(titanic_test$embarked == "Q", 1, 0)
knn_test$embarked <- NULL

set.seed(1)

parametergitter = 
  expand.grid(.k = c(1:10))

knn_modell <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarkedC+embarkedQ+embarkedS, 
      data=knn_train, method="knn",
      metric="Accuracy",
      tuneGrid=parametergitter,
      na.action = na.omit)

knn_pred <- predict(knn_modell, knn_test)
confusionMatrix(knn_pred, knn_test$survived)

knn_pred_prob <- predict(knn_modell, knn_test, type = "prob")

  # Naive Bayes 
set.seed(1)

nb_modell <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, 
                    data= titanic_train, method="nb",
                    metric = "Accuracy",
                    na.action = na.omit)

nb_pred <- predict(nb_modell, titanic_test)
confusionMatrix(nb_pred, knn_test$survived)

nb_pred_prob <- predict(nb_modell, titanic_test, type = "prob")

#SVM 
set.seed(1)

svm_modell_1 <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarkedC+embarkedQ+embarkedS, 
                      data=knn_train, method= "svmRadialSigma",
                      metric="Accuracy",
                      na.action = na.omit)
svm_pred <- predict(svm_modell_1, knn_test)

confusionMatrix(svm_pred, knn_test$survived)

  #Class probabilities
train2 <- knn_train %>% mutate(survived = if_else(survived == 1, "yes", "no"))
test2 <- knn_test%>% mutate(survived = if_else(survived == 1, "yes", "no"))
train2$survived <- as.factor(train2$survived)
test2$survived <- as.factor(test2$survived)

svm_modell_yes <-  train(survived ~ pclass+sex+age+sibsp+parch+fare+embarkedC+embarkedQ+embarkedS, 
                    data=train2, method= "svmRadialSigma",
                    metric="Accuracy",
                    na.action = na.omit, 
                    trControl = trainControl(classProbs =  TRUE))

svm_pred_prob <- predict(svm_modell_yes, test2, type = "prob")

####### Caret-Package für alle Modelle #######

##a) Random Forest
set.seed(1)
rf_modell_caret <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, 
                    data=titanic_train, method="rf",
                    metric="Accuracy",
                    na.action = na.exclude)

rf_pred_caret <- predict(rf_modell_caret, titanic_test)
confusionMatrix(rf_pred_caret, titanic_test$survived)

rf_pred_caret_prob <- predict(rf_modell_caret, titanic_test, type = "prob")

##b) GBM 
set.seed(1)
gbm_modell_caret <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, 
                         data=titanic_train, method="gbm",
                         metric="Accuracy",
                         na.action = na.exclude)

gbm_pred_caret <-predict(gbm_modell_caret, titanic_test)
confusionMatrix(gbm_pred_caret, titanic_test$survived)

gbm_pred_caret_prob <-predict(gbm_modell_caret, titanic_test, type = "prob")

##c) Logit
set.seed(1)
logit_modell_caret <- train(survived ~ pclass+sex+age+sibsp+parch+fare+embarked, 
                            data=titanic_train, method="glm",
                            family = "binomial",
                            metric="Accuracy",
                            na.action = na.exclude)
logit_pred_caret <- predict(logit_modell_caret, titanic_test)
confusionMatrix(logit_pred_caret, titanic_test$survived)

logit_pred_caret_prob <- predict(logit_modell_caret, titanic_test, type = "prob")

# Vergleichen Performances der Modelle
knn <- auc(knn_test$survived, knn_pred)

nb <- auc(titanic_test$survived, nb_pred)

svm <- auc(knn_test$survived, svm_pred)

rf_caret <- auc(titanic_test$survived, rf_pred_caret)

gbm_caret <- auc(titanic_test$survived, gbm_pred_caret)

logit_caret <- auc(titanic_test$survived, logit_pred_caret)

sprintf("k-NN AUC: %.3f", knn)
sprintf("Naives Bayes: %.3f", nb)
sprintf("SVM: %.3f", svm)
sprintf("Random Forest: %.3f", rf_caret)
sprintf("GBM: %.3f", gbm_caret)
sprintf("Logit: %.3f", logit_caret)

# Plot all models

library(ROCR)

preds_list <- list(knn_pred_prob[,1], nb_pred_prob[,1], svm_pred_prob[,1])
m <- length(preds_list)
actuals_list <- rep(list(titanic_test$survived),m)

# Plot the ROC curves 
pred <- prediction(preds_list, actuals_list)

rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("k-NN", "Naive Bayes", "SVM"),
       fill = 1:m)

#Vergleich insgesamt (mit RF, GBM und Logit)
preds_list <- list(logit_pred_caret, rf_pred_caret_prob[,1], gbm_pred_caret, knn_pred_prob[,1], nb_pred_prob[,1], svm_pred_prob[,1])
m <- length(preds_list)
actuals_list <- rep(list(titanic_test$survived),m)

# Plot the ROC curves 
pred <- prediction(preds_list, actuals_list)

rocs <- performance(pred, "tpr", "fpr")
plot(rocs, col = as.list(1:m), main = "Test Set ROC Curves")
legend(x = "bottomright", 
       legend = c("Logit", "Random Forest","GBM","k-NN", "Naive Bayes", "SVM"),
       fill = 1:m)





