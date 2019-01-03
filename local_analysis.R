library(tidyverse)
library(ggplot2)
library(glmnet)
library(randomForest)
library(betareg)
library(flexmix)


# supporting functions
logit <- function(p){
  log(p / (1 - p))
}

expit <- function(x){
  exp(x) / (1 + exp(x))
}

zero_one_scale <- function(x, n, prior = 0.5){
  (x * (n - 1) + prior) / n
}

mse <- function(pred, truth){
  mean((pred - truth)^2)
}

mae <- function(pred, truth){
  mean(abs(pred - truth))
}

# set data directory
setwd('C:/Users/fredl/Documents/repos/hnscc-image-analysis/data')

# load datasets
train_dat <- read.csv('train_dat.csv')
dev_dat <- read.csv('dev_dat.csv')
train_dat <- train_dat %>% select(-c('slide','n_cells_corrected'))
dev_dat <- dev_dat %>% select(-c('slide','n_cells_corrected'))


mod1 <- lm(y ~ ., data=train_dat)
summary(mod1)
preds <- predict(mod1, newdata=dev_dat)
mse(preds, dev_dat$y)
cor(preds, dev_dat$y)


mod2 <- glm(cbind(y*tumor_150,(1-y)*tumor_150) ~ ., family=binomial, data=train_dat)
summary(mod2)
preds <- predict(mod2, newdata=dev_dat, type='response')
mse(preds, dev_dat$y)
cor(preds, dev_dat$y)
mae(preds, dev_dat$y)


mod3 <- betareg(zero_one_scale(y, n=1000)~., data=train_dat)
summary(mod3)
preds <- predict(mod3, newdata=dev_dat)
mse(preds, dev_dat$y)
cor(preds, dev_dat$y)
mae(preds, dev_dat$y)


mod4 <- randomForest(y~., data=train_dat, ntree=300, mtry=8)
preds <- predict(mod4, newdata=dev_dat)
mse(preds, dev_dat$y)
cor(preds, dev_dat$y)
mae(preds, dev_dat$y)


mod5 <- flexmix(cbind(y*tumor_150,(1-y)*tumor_150)~., data=train_dat, k=2, 
                model=FLXMRglm(family='binomial'))
preds <- data.frame(predict(mod5, newdata=dev_dat))
cluster_true <- clusters(mod5, dev_dat)

final_pred <- vector()
for (i in 1:nrow(dev_dat)){
  final_pred[i] <- preds[i, cluster_true[i]]
}

mse(final_pred, dev_dat$y)
cor(final_pred, dev_dat$y)
mae(final_pred, dev_dat$y)


# can we predict those clusters?
training_clusters <- clusters(mod5, train_dat)
X_train <- train_dat %>% select(-'y')
X_dev <- dev_dat %>% select(-'y')

clf <- randomForest(as.factor(training_clusters) ~ ., data=X_train, ntree=300)
bestmtry <- tuneRF(X_train, as.factor(training_clusters), stepFactor=1.5, improve=1e-5, ntree=300)
print(bestmtry)
cluster_preds <- as.numeric(predict(clf, X_dev))
table(cluster_preds, cluster_true)


final_pred <- vector()
for (i in 1:nrow(dev_dat)){
  final_pred[i] <- preds[i, cluster_preds[i]]
}
mse(final_pred, dev_dat$y)
cor(final_pred, dev_dat$y)
mae(final_pred, dev_dat$y)
