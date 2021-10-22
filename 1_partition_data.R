library(fs)
library(groupdata2)
library(janitor)
library(readxl)
library(tidyverse)
library(tools)


set_100 <- read_csv('./data/set_100.csv') %>%
    select(subject_id)

segmentations <- data.frame(posterior = list.files('/Volumes/External/datasets/i-ROP/segmentations/'))

data_irop <- read_xlsx('./data/6408posterior1202irop.xlsx') %>%
    clean_names() %>%
    rename(plus = golden_reading_plus) %>%
    filter(!str_detect(subject_id, 'APEC'),
           plus != 'Unknown') %>%
    separate(subject_id, c('site', 'id')) %>%
    mutate(subject_id = paste(toupper(site), id, sep = '-'),
           plus = case_when(plus == 'No' ~ 1,
                            plus == 'Pre-Plus' ~ 2,
                            plus == 'Plus' ~ 3),
           posterior = paste(basename(file_path_sans_ext(posterior)), 'bmp', sep = '.'),
           across(c('subject_id', 'plus'), as.factor)) %>%
    inner_join(segmentations, by = 'posterior') %>%
    anti_join(set_100, by = 'subject_id') %>%
    select(subject_id, plus, posterior)


set.seed(1337)
partitioned_data <- partition(data_irop,
                              p = c(0.6, 0.2),
                              id_col = 'subject_id')

train_data <- partitioned_data[[1]]
val_data <- partitioned_data[[2]]
test_data <- partitioned_data[[3]]


pgan_no <- train_data %>%
    filter(plus == 1)

pgan_preplus <- train_data %>%
    filter(plus == 2)

pgan_plus <- train_data %>%
    filter(plus == 3)


dir_create('./data/train/real/1')
dir_create('./data/train/real/2')
dir_create('./data/train/real/3')

dir_create('./data/train/synthetic/1')
dir_create('./data/train/synthetic/2')
dir_create('./data/train/synthetic/3')

dir_create('./data/val/1')
dir_create('./data/val/2')
dir_create('./data/val/3')

dir_create('./data/test/1')
dir_create('./data/test/2')
dir_create('./data/test/3')


write_csv(train_data, './data/train.csv')
write_csv(val_data, './data/val.csv')
write_csv(test_data, './data/test.csv')


src = '/Volumes/External/datasets/i-ROP/segmentations'

dst = './data/train/real'
file_copy(paste(src, train_data$posterior, sep = '/'),
          paste(dst, train_data$plus, train_data$posterior, sep = '/'),
          overwrite = TRUE)

dst = './data/val'
file_copy(paste(src, val_data$posterior, sep = '/'),
          paste(dst, val_data$plus, val_data$posterior, sep = '/'),
          overwrite = TRUE)

dst = './data/test'
file_copy(paste(src, test_data$posterior, sep = '/'),
          paste(dst, test_data$plus, test_data$posterior, sep = '/'),
          overwrite = TRUE)
