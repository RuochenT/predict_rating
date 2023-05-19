
# libraries
import torch
import pandas as pd
import evaluate
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoConfig
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification, TrainerCallback
from datasets import load_dataset, ClassLabel, Value, load_metric, load_from_disk,ReadInstruction

#-------------- import data set in DatasetDict format and split train/validation 70:30

df_train= load_dataset("csv", data_files="/new_df_train.csv",
                       split="train") # should define split
df_valid = load_dataset("csv", data_files="/new_df_valid.csv",
                       split="train")
print(df_train)

# remove unuseful columns (diff format from pandas)
train = df_train.remove_columns(['Hotel_Name', 'Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date', 'Average_Score', \
                      'Reviewer_Nationality', 'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given'
                                 , "Tags","days_since_review","Positive_Review","Negative_Review","lat","lng","review_type","customerID"])
valid = df_valid.remove_columns(['Hotel_Name', 'Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date', 'Average_Score', \
                      'Reviewer_Nationality', 'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given'
                                 , "Tags","days_since_review", "Positive_Review","Negative_Review","lat","lng","review_type","customerID"])


#-------------- BERT encoding
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_tokenized = train.map(lambda batch: tokenizer(batch['review'], padding='max_length', truncation=True, max_length=32))
valid_tokenized = valid.map(lambda batch: tokenizer(batch['review'], padding='max_length', truncation=True, max_length=32))
df_tokenized= df.map(lambda batch: tokenizer(batch['review'], padding='max_length', truncation=True, max_length=32))


valid_tokenized = valid_tokenized.rename_column("Reviewer_Score", "labels")
valid_tokenized = valid_tokenized.rename_column("review", "text")

train_tokenized = train_tokenized.rename_column("Reviewer_Score", "labels")
train_tokenized = train_tokenized.rename_column("review", "text")



valid_tokenized.set_format("torch", columns=["input_ids",
                                             "attention_mask",
                                             "labels","token_type_ids"])
train_tokenized.set_format("torch", columns=["input_ids",
                                             "attention_mask",
                                             "labels","token_type_ids"])

train_tokenized[0] # check
valid_tokenized[0]

train_tokenized.save_to_disk("train_tokenized")
valid_tokenized.save_to_disk("valid_tokenized")

# ---------------------------------- load data set
train_tokenized = load_from_disk("/train_tokenized")
valid_tokenized = load_from_disk("/valid_tokenized")
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')


# dynmaic padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#--------------- fine tuning
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def compute_metrics_for_regression(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    mse = mean_squared_error(labels, logits)
    rmse = mean_squared_error(labels, logits, squared=False)
    mae = mean_absolute_error(labels, logits)
    r2 = r2_score(labels, logits)
    smape = 1/len(labels) * np.sum(2 * np.abs(logits-labels) / (np.abs(labels) + np.abs(logits))*100)

    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "smape": smape}

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)


training_args = TrainingArguments(
    evaluation_strategy = "epoch", # show eval loss in each epoch
    logging_strategy="epoch", # show validation?
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    metric_for_best_model = 'rmse',
    weight_decay=0.01,
    seed = 1998,
     output_dir="/Users/ruochentan1/PycharmProjects/pythonProject1/")

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_tokenized,
    eval_dataset= valid_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_for_regression
)

import copy
class CustomCallback(TrainerCallback):

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = copy.deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy
trainer.add_callback(CustomCallback(trainer))
trainer.train()

trainer.state.log_history # sum all history of training loss and metrics

trainer.evaluate() # call the summary


#----------- save fine tuned models
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = "/Users/ruochentan1/PycharmProjects/pythonProject1"

# Step 1: Save a model, configuration and vocabulary that you have fine-tuned

# If we have a distributed model, save only the encapsulated model
# (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
model_to_save = model.module if hasattr(model, 'module') else model

# If we save using the predefined names, we can load using `from_pretrained`
import os
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)

torch.save(model_to_save.state_dict(), output_model_file)
model_to_save.config.to_json_file(output_config_file)
tokenizer.save_vocabulary(output_dir)

# Step 2: Re-load the saved model and vocabulary

# Example for a Bert model
output_dir = "/Users/ruochentan1/PycharmProjects/pythonProject1"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed


#--------------------- predict on validation dataset
import evaluate
predictions = trainer.predict(valid_tokenized)
metric = evaluate.load("mse")
# transform logits to compare with the label from validation set
import numpy as np
metric.compute(predictions=predictions.predictions, references=predictions.label_ids) #regression no need to change output
# mse = 1.54

# --------- predict with test set
# load and preprocessing
df_test = load_dataset("csv", data_files="/new_df_test.csv",
                       split="train")
test_tokenized = df_test.map(lambda batch: tokenizer(batch['review'], padding='max_length', truncation=True, max_length=32))
test_tokenzied = test_tokenized.remove_columns(['Hotel_Name', 'Hotel_Address', 'Additional_Number_of_Scoring', 'Review_Date', 'Average_Score', \
                      'Reviewer_Nationality', 'Review_Total_Negative_Word_Counts', 'Total_Number_of_Reviews', 'Review_Total_Positive_Word_Counts', 'Total_Number_of_Reviews_Reviewer_Has_Given'
                                 , "Tags","days_since_review"])
test_tokenized = test_tokenized.rename_column("Reviewer_Score", "labels")
test_tokenized = test_tokenized.rename_column("review", "text")
test_tokenized.set_format("torch", columns=["input_ids", "token_type_ids", "attention_mask", "labels"])

test_tokenized.save_to_disk("test_tokenized")

# predict with fine-tuned model
output_dir = "/pythonProject1"
model = BertForSequenceClassification.from_pretrained(output_dir)
tokenizer = BertTokenizer.from_pretrained(output_dir)  # Add specific options if needed
test_tokenized = load_from_disk("/test_tokenized")
training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    logging_strategy= "epoch",
    num_train_epochs=3,
    weight_decay=0.01,
     output_dir="/pythonProject1/")

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    tokenizer=tokenizer
)

predictions2 = trainer.predict(test_tokenized) #predictions2.predictions has all value but label_ids include Nan
# metric.compute(predictions=predictions2.predictions, references=predictions2.label_ids) # input contains NAN cannot compute


#---- comparison with original labels
df_test = pd.read_csv("/Users/ruochentan1/PycharmProjects/pythonProject1/new_df_test.csv")
df_test_full= pd.read_csv("/Users/ruochentan1/PycharmProjects/pythonProject1/new_df_test_full.csv")
df_test["Fine-tuned_Score"] = predictions2.predictions
df_test["Full_Score"] =df_test_full["Reviewer_Score"]
df_test["added_score"]=df_test['Reviewer_Score'].fillna(df_test["Fine-tuned_Score"]) # add fine-tuned score to nan in reviewer score


df_test.to_csv("new_df_test_result.csv", index =False, header = True)


