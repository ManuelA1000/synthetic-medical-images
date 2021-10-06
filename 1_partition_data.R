library(tidyverse)
library(janitor)
library(readxl)
library(tools)
library(fs)



read_split <- function(file, sheet, join_with) {
    split <- read_excel(file, sheet = sheet) %>%
        clean_names() %>%
        separate(original_files, c('subject_id', NA, 'session', NA, 'eye', NA), sep = '_') %>%
        inner_join(join_with, by = c('subject_id', 'eye', 'session')) %>%
        na_if('NULL.bmp') %>%
        drop_na() %>%
        select(subject_id, session, eye, ground_truth, posterior)
    
    split
}



data_irop <- read_csv('/Volumes/External/irop_data/irop_07092020.csv') %>%
    clean_names() %>%
    filter(reader == 'goldenstandardreading@ohsu.edu',
           !str_detect(subject_id, 'APEC')) %>%
    mutate(session = str_remove_all(session, '[ a-zA-Z]'),
           posterior = paste(file_path_sans_ext(basename(posterior)), '.bmp', sep = '')) %>%
    select(subject_id, eye, session, posterior)

split_file = '/Volumes/External/irop_data/all_splits_master_file.xlsx'
split_1 <- read_split(split_file, sheet = 1, join_with = data_irop)
split_2 <- read_split(split_file, sheet = 2, join_with = data_irop)
split_3 <- read_split(split_file, sheet = 3, join_with = data_irop)
split_4 <- read_split(split_file, sheet = 4, join_with = data_irop)
split_5 <- read_split(split_file, sheet = 5, join_with = data_irop)

data_train <- bind_rows(split_1, split_2, split_3)
data_val <- split_4
data_test <- split_5

pgan_no <- data_train %>%
    filter(ground_truth == 'No')

pgan_preplus <- data_train %>%
    filter(ground_truth == 'Pre-Plus')

pgan_plus <- data_train %>%
    filter(ground_truth == 'Plus')


dir_create('./out/train/real/No')
dir_create('./out/train/real/Pre-Plus')
dir_create('./out/train/real/Plus')

dir_create('./out/train/synthetic/No')
dir_create('./out/train/synthetic/Pre-Plus')
dir_create('./out/train/synthetic/Plus')

dir_create('./out/val/No')
dir_create('./out/val/Pre-Plus')
dir_create('./out/val/Plus')

dir_create('./out/test/No')
dir_create('./out/test/Pre-Plus')
dir_create('./out/test/Plus')


write_csv(data_train, './out/train/cnn_train.csv')
write_csv(data_val, './out/val/cnn_val.csv')
write_csv(data_test, './out/test/cnn_test.csv')


src = '/Volumes/External/irop_data/segmentations'

dst = './out/train/real'
file_copy(paste(src, data_train$posterior, sep = '/'),
          paste(dst, data_train$ground_truth, data_train$posterior, sep = '/'),
          overwrite = TRUE)

dst = './out/val'
file_copy(paste(src, data_val$posterior, sep = '/'),
          paste(dst, data_val$ground_truth, data_val$posterior, sep = '/'),
          overwrite = TRUE)

dst = './out/test'
file_copy(paste(src, data_test$posterior, sep = '/'),
          paste(dst, data_test$ground_truth, data_test$posterior, sep = '/'),
          overwrite = TRUE)

file_move('./out/train/real/No', './out/train/real/1')
file_move('./out/train/real/Pre-Plus', './out/train/real/2')
file_move('./out/train/real/Plus', './out/train/real/3')

file_move('./out/val/No', './out/val/1')
file_move('./out/val/Pre-Plus', './out/val/2')
file_move('./out/val/Plus', './out/val/3')

file_move('./out/test/No', './out/test/1')
file_move('./out/test/Pre-Plus', './out/test/2')
file_move('./out/test/Plus', './out/test/3')
