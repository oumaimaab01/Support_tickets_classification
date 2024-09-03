import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

# Charger le dataset

dfTickets = pd.read_csv('./all_tickets.csv', dtype=str)

# Paramètres
column_to_predict = "ticket_type"
text_columns = "body"

# Préparer les données

labelData = dfTickets[column_to_predict].astype('category').cat.codes
data = dfTickets[text_columns].fillna("")

# Split dataset into training and testing data
train_texts, test_texts, train_labels, test_labels = train_test_split(data, labelData, test_size=0.2, random_state=42)
# Tokenizer BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Fonction pour tokenizer les textes

def tokenize_function(texts):
    return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=512)

train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# Créer un Dataset PyTorch

class TicketDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TicketDataset(train_encodings, train_labels.values)
test_dataset = TicketDataset(test_encodings, test_labels.values)

# Charger le modèle BERT pré-entraîné

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dfTickets[column_to_predict].unique()))

# Définir les arguments d'entraînement

training_args = TrainingArguments(
output_dir='./results',
num_train_epochs=3,
per_device_train_batch_size=16,
per_device_eval_batch_size=64,
warmup_steps=500,
weight_decay=0.01,
logging_dir='./logs',
logging_steps=10,
evaluation_strategy="epoch",
)

# Utiliser le Trainer de Hugging Face pour l'entraînement

trainer = Trainer(
model=model,
args=training_args,
train_dataset=train_dataset,
eval_dataset=test_dataset,
)

# Entraîner le modèle

trainer.train()

# Sauvegarder le modèle et le tokenizer

model_save_path = "./saved_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")

# Évaluer le modèle

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

# Afficher les résultats

print("Classification Report:")
print(classification_report(test_labels, preds, target_names=dfTickets[column_to_predict].astype('category').cat.categories))

# Matrice de confusion

cm = confusion_matrix(test_labels, preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=dfTickets[column_to_predict].astype('category').cat.categories, yticklabels=dfTickets[column_to_predict].astype('category').cat.categories)
plt.xlabel('True Label')
plt.ylabel('Predicted Label')
plt.show()