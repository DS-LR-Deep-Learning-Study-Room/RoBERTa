PROJECT_NAME = "RoBERTa"

### URL
PRETRAIN_URL = "FacebookAI/xlm-roberta-base"
MODEL_PATH = PROJECT_NAME + "/model/roberta.pt"
LOGGING_PATH = PROJECT_NAME + "/logs"
TRAINER_PATH = PROJECT_NAME + "/trainer"

### Dataset
TRAIN_SET = "train.parquet"
TEST_SET = "test.parquet"
VALID_SET = "valid.parquet"
