import torch
import torch.nn as nn
import numpy as np

# add the path to my packages to system paths so they can be imported
import sys
# sys.path.append('/home/yasamanparhizkar/Documents/yorku/01_thesis/simgraph/code/my_packages')
sys.path.append('F:/Users/yasam/Documents/GitHub/simgraph/code/my_packages')
# sys.path.append('/home/yasamanparhizkar/Documents/thesis/code/my_packages')

import dataprocess.data_handler_03 as dh

# Fully connected neural network with one hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        # -> x needs to be: (batch_size, seq, input_size)
        
        # or:
        #self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x, device='cpu'):
        # Set initial hidden states (and cell states for LSTM)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        #c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        
        # x: (n, 28, 28), h0: (2, n, 128)
        
        # Forward propagate RNN
        out, _ = self.rnn(x, h0)  
        # or:
        #out, _ = self.lstm(x, (h0,c0))  
        
        # out: tensor of shape (batch_size, seq_length, hidden_size)
        # out: (n, 28, 128)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        # out: (n, 128)
         
        out = self.fc(out)
        # out: (n, 10)
        return out

def prepare_rnn_data(data_params, train_num, val_num, batch_size_train, batch_size_val, 
                     sequence_length, input_size, seed=1342):
    train_num, val_num, train_data, val_data = \
    dh.random_train_val_balanced(train_num, val_num, data_params, seed=seed)
    
    train_des = train_data['des'].reshape(-1, batch_size_train, sequence_length, input_size)
    train_lbls = train_data['lbls'].reshape(-1, batch_size_train)
    train_lbls = (train_lbls + 1)//2

    val_des = val_data['des'].reshape(-1, batch_size_val, sequence_length, input_size)
    val_lbls = val_data['lbls'].reshape(-1, batch_size_val)
    val_lbls = (val_lbls + 1)//2
    
    # number of all batches in the training data
    n_total_steps = train_num // batch_size_train
    n_val_batches = val_num // batch_size_val
    # print('Should show 1, 1: ', n_total_steps, ',', n_val_batches)
    
    return train_des, train_lbls, val_des, val_lbls, n_total_steps, n_val_batches

def get_datapoint(des, lbls, step_i):
    features = torch.from_numpy(des[step_i, :, :, :]).to(torch.float32)
    labels = torch.from_numpy(lbls[step_i, :]).to(torch.int64)
    
    return features, labels

def train_rnn(input_size, hidden_size, num_layers, num_classes, device, learning_rate, 
              num_epochs, n_total_steps, train_des, train_lbls, show_messages=False):
    model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    for epoch in range(num_epochs):
        for i in range(n_total_steps):  
            # origin shape: [N, 1, 28, 28]
            # resized: [N, 28, 28]
            images, labels = get_datapoint(train_des, train_lbls, i)

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images, device)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if show_messages and (i+1) % 1 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
                
    return model

def test_rnn(model, n_val_batches, val_des, val_lbls, device='cpu'):
    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i in range(n_val_batches): # take care not to repeat training data
            images, labels = get_datapoint(val_des, val_lbls, i)
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, device)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples 
    
    return acc


def assess_rnn(train_sizes, val_num, train_its, val_its, data_params, sequence_length, input_size, hidden_size, num_layers, num_classes, device, learning_rate, num_epochs,show_messages=False, seed=None):
    
    val_acc_means = []
    val_acc_stds = []

    for train_num in train_sizes:
        it_accs = []
        for tr_it in range(train_its):
            # prepare the training data
            # to resemble the graph optimization better, we only have one large batch.
            batch_size_train = train_num
            batch_size_val = val_num
            train_des, train_lbls, _, _, n_total_steps, _ = \
            prepare_rnn_data(data_params, train_num, val_num, batch_size_train, batch_size_val, 
                                 sequence_length, input_size, seed)

            # train
            model = train_rnn(input_size, hidden_size, num_layers, num_classes, device, learning_rate, 
                  num_epochs, n_total_steps, train_des, train_lbls, show_messages)

            for va_it in range(val_its):
                # prepare the validation data
                _, _, val_des, val_lbls, _, n_val_batches = \
                prepare_rnn_data(data_params, train_num, val_num, batch_size_train, batch_size_val, 
                                     sequence_length, input_size, seed)

                # test
                acc = test_rnn(model, n_val_batches, val_des, val_lbls, device)
                it_accs.append(acc)
                # print(f'Accuracy of the network on the test images: {acc} %')
            print('train size: {} iteration: {}/{}'.format(train_num, tr_it+1, train_its))

        val_acc_means.append(np.mean(it_accs))
        val_acc_stds.append(np.std(it_accs))
        print('train size: {} \tmean acc: {} \tstd: {}'.format(train_num, val_acc_means[-1], val_acc_stds[-1]))
        
    return val_acc_means, val_acc_stds