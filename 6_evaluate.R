library(here)
library(tidyverse)
library(ROCR)
library(pROC)
library(caret)
library(boot)
library(umap)
library(ggthemes)

## set seed for repeatability
set.seed(1337)


## create helper functions
read_predictions <- function(dataset) {
    read_csv(dataset) %>%
        mutate(label = factor(label, labels = c('Normal', 'Pre-Plus', 'Plus')),
               prediction = factor(prediction, labels = c('Normal', 'Pre-Plus', 'Plus')),
               is_plus = if_else(label == 'Plus', 'Plus', 'No')) %>%
        rename('p_no' = '1', 'p_pp' = '2', 'p_plus' = '3')
}


read_features <- function(dataset) {
    read_csv(dataset) %>%
        rename('image_path' = '0') %>%
        separate(image_path, c(NA, NA, NA, NA, 'label', NA), remove = FALSE, sep = '/')
}


plot_roc <- function(dataset, save_name) {
    png(save_name, width = 7, height = 5, units = 'in', res = 300)
    pred <- prediction(dataset['p_plus'], dataset['is_plus'])
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    auc <- performance(pred, measure = "auc")
    auc <- auc@y.values[[1]]
    title <- paste('Real Images - ROC: Plus vs Normal/Pre-plus [AUC = ', round(auc, 3), ']', sep = '')
    plot(perf, main = title, lwd=2)
    abline(a= 0, b=1, col = 'red', lty = 'dashed', lwd=2)
}


plot_pr <- function(dataset, save_name) {
    png(save_name, width = 7, height = 5, units = 'in', res = 300)
    num_total <- nrow(dataset)
    num_plus <- nrow(filter(dataset, label == 'Plus'))
    pred <- prediction(dataset['p_plus'], dataset['is_plus'])
    perf <- performance(pred, measure = "prec", x.measure = "rec")
    auc <- performance(pred, measure = "aucpr")
    auc <- auc@y.values[[1]]
    title <- paste('PR: Plus vs Normal/Pre-plus [AUC = ', round(auc, 3), ']', sep = '')
    plot(perf, main = title , ylim=c(0,1), lwd=2)
    abline(a = num_plus/num_total, b = 0, col = 'red', lty = 'dashed', lwd=2)
}


boot_auc_pr <- function(dataset, indices) {
    d <- dataset[indices,]
    pred <- prediction(d['p_plus'], d['is_plus'])
    auc <- performance(pred, measure = "aucpr")
    auc <- auc@y.values[[1]]
    auc
}


boot_auc_roc <- function(dataset, indices) {
    d <- dataset[indices,]
    pred <- prediction(d['p_plus'], d['is_plus'])
    auc <- performance(pred, measure = "auc")
    auc <- auc@y.values[[1]]
    auc
}


## load data
real_test <- read_predictions(here('out', 'cnn', 'real_test_data_probabilities.csv'))
real_set_100 <- read_predictions(here('out', 'cnn', 'real_set_100_probabilities.csv'))

synth_test <- read_predictions(here('out', 'cnn', 'synthetic_test_data_probabilities.csv'))
synth_set_100 <- read_predictions(here('out', 'cnn', 'synthetic_set_100_probabilities.csv'))

real_features <- read_features(here('out', 'cnn', 'real_features.csv'))
synth_features <- read_features(here('out', 'cnn', 'synthetic_features.csv'))


## assess precision-recall
plot_pr(real_test, here('out', 'cnn', 'real_pr.png'))
plot_pr(synth_test, here('out', 'cnn', 'synth_pr.png'))

real_boot <- boot(data = real_test, statistic = boot_auc_pr, R = 1000, sim = 'ordinary')
real_boot
boot.ci(real_boot, type = "perc")

synth_boot <- boot(data = synth_test, statistic = boot_auc_pr, R = 1000, sim = 'ordinary')
synth_boot
boot.ci(synth_boot, type = "perc")

(sum(real_boot$t > synth_boot$t0) + 1) / (1000 + 1)


## assess receiver operating characteristics
plot_roc(real_test, here('out', 'cnn', 'real_pr.png'))
plot_roc(synth_test, here('out', 'cnn', 'synth_pr.png'))

real_boot <- boot(data = real_test, statistic = boot_auc_roc, R = 1000, sim = 'ordinary')
real_boot
boot.ci(real_boot, type = "perc")

synth_boot <- boot(data = synth_test, statistic = boot_auc_roc, R = 1000, sim = 'ordinary')
synth_boot
boot.ci(synth_boot, type = "perc")

(sum(real_boot$t > synth_boot$t0) + 1) / (1001)

roc.test(real_test$is_plus, real_test$p_plus, synth_test$p_plus, method = 'delong')


## assess expert test set performance
confusionMatrix(real_set_100$prediction, real_set_100$label)
confusion_matrices = array(c(54, 0, 0, 0, 31, 0, 0, 0, 15,
                             54, 0, 0, 8, 23, 0, 1, 3, 11),
                           dim=c(3, 3, 2))
mantelhaen.test(confusion_matrices)

confusionMatrix(synth_set_100$prediction, synth_set_100$label)
confusion_matrices = array(c(54, 0, 0, 0, 31, 0, 0, 0, 15,
                             52, 2, 0, 4, 19, 8, 0, 1, 14),
                           dim=c(3, 3, 2))
mantelhaen.test(confusion_matrices)


## create UMAP embedding
config = umap.defaults
config$random_state = 1337
config$min_dist = 0.99
config$metric = 'euclidean'

train_umap <- umap(real_features[,3:ncol(real_features)], config = config)

synth_pred_fts <- synth_features %>%
    select(-image_path, -label) %>%
    predict(train_umap, .)

real_ft_pred <- real_features %>%
    select(label) %>%
    mutate(color = case_when(label == 1 ~ '#3e5629',
                             label == 2 ~ 'darkorange4',
                             label == 3 ~ 'red4')) %>%
    cbind(train_umap$layout)

synth_features %>%
    select(label) %>%
    mutate(color = case_when(label == 1 ~ '#73aa54',
                             label == 2 ~ '#df8244',
                             label == 3 ~ '#d84c54')) %>%
    cbind(synth_pred_fts) %>%
    ggplot() +
        aes(x = `1`, y = `2`) +
        geom_point(aes(color = color), shape = 16, alpha = 0.5, size = 2) +
        geom_point(data = real_ft_pred, aes(color = color), shape = 17, size = 1.5) +
        scale_color_identity() +
        theme_base() +
        scale_x_continuous(name = 'Vector 1', breaks = c(seq(-10, 10, 4))) +
        scale_y_continuous(name = 'Vector 2', breaks = c(seq(-10, 10, 4)))

ggsave('./out/umap/umap.png')

write_csv(as.data.frame(train_umap$layout), here('out', 'umap', 'train_layout.csv'))
write_csv(as.data.frame(synth_pred_fts), here('out', 'umap', 'synth_layout.csv'))
