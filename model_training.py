import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.F1_score_running import F1_score_running

import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='data', help='Directory containing the dataset')
parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the model')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')

args = parser.parse_args()

class LRCNModel(nn.Module):
    def __init__(self, conv_channels=16, hidden_size=128, num_layers=1, num_classes=3):
        super(LRCNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv_channels,
                             kernel_size=3, padding="same")
        self.dp1 = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels,
                               kernel_size=3, padding="same")
        self.dp2 = nn.Dropout(0.5)
        self.lstm = nn.LSTM(input_size=conv_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, 1, 852)
        x = self.conv1(x)
        x = self.dp1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.dp2(x)
        x = self.relu(x)
        # x shape: (batch_size, conv_channels, 852)
        x = x.permute(0, 2, 1)  # Shape to (batch_size, seq_len=852, input_size=conv_channels)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Last time step
        x = self.fc(x)
        return x

# Hyperparameters
num_epochs = 10
batch_size = 32
learning_rate = 0.001

# Initialize model, loss function, optimizer
model = LRCNModel()
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

class SynthDataset(Dataset):
    def __init__(self, data_dir, dt_type = None):
        assert dt_type in ['Train', 'Val', 'Test']
        self.data_dir = os.path.join(data_dir, dt_type)
        self.data = os.listdir(self.data_dir)
        self.labels = []
        for input in self.data:
            self.labels.append(int(input.split('_')[1].split('.')[0][-1]))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(np.load(os.path.join(self.data_dir,self.data[idx]))).float(), torch.tensor(self.labels[idx])

train_dl = DataLoader(SynthDataset(args.data_dir, dt_type = 'Train'),
                       batch_size=batch_size, shuffle=True)
val_dl = DataLoader(SynthDataset(args.data_dir, dt_type = 'Val'),
                          batch_size=batch_size, shuffle=True)
test_dl = DataLoader(SynthDataset(args.data_dir, dt_type = 'Test'),
                          batch_size=batch_size, shuffle=True)

# Training loop
model.train()
model= model.float()
epoch_number = 0
for epoch in tqdm(range(args.num_epochs), f"Epoch:{epoch_number}/{args.num_epochs}"):

    model.train()
    avg_loss = 0
    avg_vloss = 0
    avg_val_accuracy = 0
    avg_train_accuracy = 0

    best_vloss = 0

    for idx, data in enumerate(train_dl):
        inputs, labels = data

        #Zeroed gradients for the optimizer
        optimizer.zero_grad()
        outputs = model(inputs)

        #Calculate loss
        loss = loss_fn(outputs, labels)
        avg_loss += loss.item()

        #Check the accuracy
        _, predicted = torch.max(outputs, 1)
        avg_train_accuracy += (predicted == labels).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    print(f'Average Train Loss: {avg_loss/(idx+1):.4f}')
    print(f'Average Train Acc: {avg_train_accuracy/(idx+1):.4f}')
    

    #Validation
    model.eval()
    with torch.no_grad():
        for idx, vdata in enumerate(val_dl):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            avg_vloss += loss_fn(voutputs, vlabels).item()

            _, predicted = torch.max(voutputs, 1)
            avg_val_accuracy += (predicted == vlabels).sum().item()

        
        print(f'Average Val Loss: {avg_vloss/(idx+1):.4f}')
        print(f'Average Val Acc: {avg_val_accuracy/(idx+1):.4f}')

    
    #Save the model if it is the best
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        torch.save(model.state_dict(), 
                   os.path.join(args.model_dir,
                                 'best_model.pth'))
        


#Test the best model
model.load_state_dict(torch.load(os.path.join(args.model_dir,
                                            'best_model.pth')))

model.eval()
test_acc = 0
f1 = F1_score_running(classes=3)
with torch.no_grad():
    for idx, tdata in enumerate(test_dl):
        tinputs, tlabels = tdata
        toutputs = model(tinputs)


        _, predicted = torch.max(toutputs, 1)
        f1.log(predicted, tlabels)
        test_acc += (predicted == tlabels).sum().item()
    
print(f'Average Test Acc: {test_acc/(idx+1):.4f}')
print(f'F1 Score: {f1.calc(average = "macro"):.4f}')

        