# Import necessary libraries
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import XLNetTokenizer, XLNetForSequenceClassification, AdamW
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Use a subset of data for quick testing
df_subset = df[:1000]  # Adjust this value as needed

# Load the XLNet tokenizer
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')

# Tokenize all of the sentences and map the tokens to their word IDs
input_ids = []
attention_masks = []

for i, review in enumerate(df_subset['review']):
    encoded_dict = tokenizer.encode_plus(
        review,
        add_special_tokens = True,
        max_length = 64,
        padding='max_length',
        truncation=True,
        return_attention_mask = True,
        return_tensors = 'pt',
    )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
    
    if (i+1) % 100 == 0:  # Adjust this value as needed
        print(f'{i+1} reviews processed')

# Convert the lists into tensors
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(df_subset['sentiment'].map({'positive': 1, 'negative': 0}).values)

# Set the batch size
batch_size = 32  # You can adjust this value as needed

# Create the DataLoader
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

# Load a pre-trained model
model = XLNetForSequenceClassification.from_pretrained("xlnet-base-cased", num_labels = 2)

# Put the model in training mode
model.train()

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Train the model for a few epochs
epochs = 4  # Replace with the number of epochs you want to train for
for epoch in range(epochs):
    for batch in prediction_dataloader:
        # Unpack the inputs from our dataloader
        input_ids, input_mask, labels = batch
        # Forward pass
        outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask, labels=labels)
        # Get the loss
        loss = outputs[0]
        # Backward pass
        loss.backward()
        # Update weights
        optimizer.step()
        optimizer.zero_grad()

# Put the model in evaluation mode
model.eval()

# Create a list to store the predictions and true labels
predictions = []
true_labels = []

for batch in prediction_dataloader:
    	# Unpack the inputs from our dataloader
    	input_ids, input_mask, labels = batch
    	# Forward pass, calculate logit predictions
    	outputs = model(input_ids, token_type_ids=None, attention_mask=input_mask)
    	# Get the "logits" output by the model. The "logits" are the output
    	# values prior to applying an activation function like the softmax.
    	logits = outputs[0]
    	# Convert logits to probabilities
    	probs = F.softmax(logits, dim=1)
    	# Get the predicted class for each review in the batch
    	pred_classes = torch.argmax(probs, dim=1)
    	# Add the predictions and true labels to our lists
    	predictions.extend(pred_classes.tolist())
    	true_labels.extend(labels.tolist())

# Calculate the accuracy of our predictions
accuracy = accuracy_score(true_labels, predictions)

print(f'Accuracy: {accuracy}')

# Calculate precision, recall, and F1 score
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Calculate the confusion matrix
cm = confusion_matrix(true_labels, predictions)
print(f'Confusion Matrix: \n{cm}')
