# libraries
import torch
import pandas as pd
import evaluate
import numpy as np
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaForSequenceClassification, TrainingArguments, Trainer,BertForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric, load_from_disk

# ---------------------------------- load data set
train_tokenized = load_from_disk("/Users/ruochentan1/PycharmProjects/pythonProject2/train_tokenized")
valid_tokenized = load_from_disk("/Users/ruochentan1/PycharmProjects/pythonProject2/valid_tokenized")
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels =1)


# dynmaic padding
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


#--------------- fine tuning
from sklearn.metrics import mean_squared_error
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


training_args = TrainingArguments(
    evaluation_strategy = "epoch", # show eval loss in each epoch
    logging_strategy="epoch", # show validation?
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=20,
    num_train_epochs=4,
    metric_for_best_model = 'rmse',
    weight_decay=0.01,
     output_dir="/Users/ruochentan1/PycharmProjects/pythonProject2/")

trainer = Trainer(
    model,
    training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics_for_regression
)

trainer.train()
trainer.state.log_history # call history of metrics and loss


trainer.evaluate() # call the summary


#----------- save fine tuned models
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = "/Users/ruochentan1/PycharmProjects/pythonProject2"

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
output_dir = "/Users/ruochentan1/PycharmProjects/pythonProject2"
model = RobertaForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Add specific options if needed


# predict with fine-tuned model
output_dir = "/Users/ruochentan1/PycharmProjects/pythonProject2"
model = RobertaForSequenceClassification.from_pretrained(output_dir)
tokenizer = AutoTokenizer.from_pretrained(output_dir)  # Add specific options if needed
test_tokenized = load_from_disk("/Users/ruochentan1/PycharmProjects/pythonProject1/test_tokenized")
training_args = TrainingArguments(
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=20,
    num_train_epochs=4,
    weight_decay=0.01,
     output_dir="/Users/ruochentan1/PycharmProjects/pythonProject2/")

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



df_test.to_csv("new_df_test_result_roberta.csv", index =False, header = True)



# Generate a sequence of integers to represent the epoch numbers
epochs = range(0, 4)
train_values = [9.8176, 4.6409, 3.6732, 3.2851]
val_values = [5.15823364, 4.36231, 4.01685,3.671488]
# Plot and label the training and validation loss values
from matplotlib.pylab import plt
import numpy as np
plt.plot(epochs, train_values, label ='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')

# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# Display the plot
plt.legend(loc='best')
plt.show()
