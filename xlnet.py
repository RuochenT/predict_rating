
# libraries
import torch
import pandas as pd
import evaluate
import numpy as np
from transformers import XLNetTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments,TrainerCallback ,Trainer,XLNetForSequenceClassification
from datasets import load_dataset, ClassLabel, Value, load_metric, load_from_disk


df_train= load_dataset("csv", data_files="/Users/ruochentan1/PycharmProjects/bert/df_train.csv",
                       split="train") # should define split
df_valid = load_dataset("csv", data_files="/Users/ruochentan1/PycharmProjects/bert/df_valid.csv",
                       split="train")

print(df_train)


# remove unuseful columns (diff format from pandas)
train = df_train.remove_columns(['Unnamed: 0', 'unixReviewTime',"reviewTime"])
valid = df_valid.remove_columns(['Unnamed: 0','unixReviewTime',"reviewTime"])

labels = [label for label in train.features.keys() if label not in ['overall', "reviewerID","asin","reviewText"]]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels

#-------------- XLnet encoding
from transformers import AutoTokenizer
import numpy as np

tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

def preprocess_data(examples):
    # take a batch of texts
    text = examples["reviewText"]
    # encode them
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=32)
    # add labels
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    # create numpy array of shape (batch_size, num_labels)
    labels_matrix = np.zeros((len(text), len(labels)))
    # fill numpy array
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]

    encoding["labels"] = labels_matrix.tolist()

    return encoding
encoded_train = train.map(preprocess_data, batched=True, remove_columns=train.column_names)
example = encoded_train[0]
print(example.keys())
encoded_valid = valid.map(preprocess_data,batched =True, remove_columns = valid.column_names)

tokenizer.decode(example['input_ids'])

encoded_train.set_format("torch")
encoded_valid.set_format("torch")

encoded_train.save_to_disk("train_tokenized")
encoded_valid.save_to_disk("valid_tokenized")

# define model
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("xlnet-base-cased",
                                                           problem_type="multi_label_classification",
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import EvalPrediction
import torch


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions,
                                           tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds,
        labels=p.label_ids)
    return result

training_args = TrainingArguments(
    evaluation_strategy = "epoch", # show eval loss in each epoch
    logging_strategy="epoch", # show validation?
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=4,
    weight_decay=0.01,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    seed = 1998,
     output_dir="/Users/ruochentan1/PycharmProjects/xlnet/")

trainer = Trainer(
    model,
    training_args,
    train_dataset=encoded_train,
    eval_dataset= encoded_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

trainer.evaluate() # evaluate on trianning dataset


#----------- save fine tuned models
from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME

output_dir = "/Users/ruochentan1/PycharmProjects/xlnet"

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
output_dir = "/Users/ruochentan1/PycharmProjects/xlnet"
model = XLNetForSequenceClassification.from_pretrained(output_dir)
tokenizer = XLNetTokenizer.from_pretrained(output_dir)  # Add specific options if needed

# predict with test dataset
encoded_train = load_from_disk("/Users/ruochentan1/PycharmProjects/xlnet/train_tokenized")
encoded_valid = load_from_disk("/Users/ruochentan1/PycharmProjects/xlnet/valid_tokenized")

df_test = load_dataset("csv", data_files="/Users/ruochentan1/PycharmProjects/bert/df_test_full.csv",
                       split="train")
test = df_test.remove_columns(['Unnamed: 0',"unixReviewTime","reviewTime"])
labels = [label for label in test.features.keys() if label not in ['overall', "reviewerID","asin","reviewText"]]
id2label = {idx:label for idx, label in enumerate(labels)}
label2id = {label:idx for idx, label in enumerate(labels)}
labels
encoded_test = test.map(preprocess_data, batched=True, remove_columns= test.column_names) # with its own labels
encoded_test.set_format("torch")
encoded_test.save_to_disk("test_tokenized")

encoded_test = load_from_disk("/Users/ruochentan1/PycharmProjects/xlnet/test_tokenized")


predictions = trainer.predict(encoded_test) # return as logits
preds = np.argmax(predictions.predictions, axis=-1) #
preds

df_test = pd.read_csv("/Users/ruochentan1/PycharmProjects/bert/df_test.csv")
df_test["preds"] = preds
df_test["preds"] = df_test.preds.replace([0, 1, 2, 3,4], [1, 2, 3, 4,5])
df_test_full = pd.read_csv("/Users/ruochentan1/PycharmProjects/bert/df_test_full.csv")
df_test["true"] = df_test_full["overall"]
df_test.to_csv("xlnet_result.csv",index =False, header = True)


from sklearn import metrics
y_pred = df_test.loc[df_test['overall'].isnull(), 'preds']
y_true = df_test.loc[df_test['overall'].isnull(),'true']
print(metrics.confusion_matrix(y_true, y_pred))
print(metrics.classification_report(y_true,y_pred,digits = 3))

from sklearn.preprocessing import LabelBinarizer

# Convert y_pred to one-hot encoded format
lb = LabelBinarizer()
y_pred_one_hot = lb.fit_transform(y_pred)

# Compute roc_auc_score
roc_auc = roc_auc_score(y_true, y_pred_one_hot, multi_class='ovr')







import matplotlib.pyplot as plt
# Generate a sequence of integers to represent the epoch numbers
epochs = range(0, 4)
train_values = [ 0.232329,  0.18058733, 0.144393,  0.12950767]
val_values = [ 0.270667, 0.2456904, 0.2645216,  0.2672869]
# Plot and label the training and validation loss values
plt.plot(epochs, train_values, label ='Training Loss')
plt.plot(epochs, val_values, label='Validation Loss')
# Add in a title and axes labels
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
# Display the plot
plt.legend(loc='best')
plt.show()