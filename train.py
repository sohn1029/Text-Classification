import argparse
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from util import MyCollate
from model import BaseModel
from vocab import Vocabulary
from torch.utils.data.sampler import SubsetRandomSampler
import pickle
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

class TextDataset(Dataset):

    def __init__ (self, data_dir, mode, vocab_size):

        self.df = pd.read_csv(os.path.join(data_dir, mode + '.csv'))

        self.sentences = self.df['text'].values
        self.labels = self.df['label'].values

        # Initialize dataset Vocabulary object and build our vocabulary
        self.sentences_vocab = Vocabulary(vocab_size)
        self.labels_vocab = Vocabulary(vocab_size)

        self.sentences_vocab.build_vocabulary(self.sentences)
        self.labels_vocab.build_vocabulary(self.labels, add_unk=False)
        
    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        
        #numericalize the sentence ex) ['cat', 'in', 'a', 'bag'] -> [2,3,9,24,22]
        numeric_sentence = self.sentences_vocab.sentence_to_numeric(sentence)
        numeric_label = self.labels_vocab.sentence_to_numeric(label)
        return torch.tensor(numeric_sentence), torch.tensor(numeric_label)


def make_data_loader(dataset, batch_size, batch_first, shuffle=True): #increase num_workers according to CPU
    #get pad_idx for collate fn
    pad_idx = dataset.sentences_vocab.wtoi['<PAD>']

    indices = list(range(len(train_dataset)))
    split = int(np.floor(0.2 * len(train_dataset)))
    np.random.seed(10)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    #define loader
    train_loader = DataLoader(dataset, batch_size = batch_size,
                        collate_fn = MyCollate(pad_idx=pad_idx, batch_first=batch_first),
                        sampler=train_sampler) #MyCollate class runs __call__ method by default
    valid_loader = DataLoader(dataset, batch_size = batch_size,
                        collate_fn = MyCollate(pad_idx=pad_idx, batch_first=batch_first),
                        sampler=valid_sampler)                    
    return train_loader, valid_loader

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, data_loader, valid_loader, model):
    """
    TODO: Change the training code as you need. (e.g. different optimizer, different loss function, etc.)
            You can add validation code. -> This will increase the accuracy.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
    min_loss = np.Inf
    
    epoch_train_acc_l = []
    epoch_valid_acc_l = []
    for epoch in range(args.num_epochs):
        train_losses = []
        valid_losses = []
        train_acc = 0.0
        valid_acc = 0.0
        total=0
        valid_total=0
        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        model.train()
        for i, (text, label) in enumerate(tqdm(data_loader)):

            text = text.to(args.device)
            label = label.to(args.device)            
            optimizer.zero_grad()

            output, _, _ = model(text)
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            total += label.size(0)

            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/total
        epoch_train_acc_l.append(epoch_train_acc)
        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))

        model.eval()
        for i, (text, label) in enumerate(tqdm(valid_loader)):

            text = text.to(args.device)
            label = label.to(args.device)            
            optimizer.zero_grad()

            output, _, _ = model(text)
            
            label = label.squeeze()
            loss = criterion(output, label)
            

            valid_losses.append(loss.item())
            valid_total += label.size(0)

            valid_acc += acc(output, label)

        epoch_valid_loss = np.mean(valid_losses)
        epoch_valid_acc = valid_acc/valid_total
        epoch_valid_acc_l.append(epoch_valid_acc)
        print(f'Epoch {epoch+1}') 
        print(f'valid_loss : {epoch_valid_loss}')
        print('valid_accuracy : {:.3f}'.format(epoch_valid_acc*100))

        # Save Model
        if epoch_valid_loss < min_loss:
            torch.save(model.state_dict(), 'best_model.pt')
            print('Train loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, epoch_valid_loss))
            min_loss = epoch_valid_loss
        torch.save(model.state_dict(), 'model.pt')
    x = range(1, args.num_epochs+1)
    plt.plot(x, epoch_train_acc_l, 'r', label = "train")
    plt.plot(x, epoch_valid_acc_l, 'b', label = "dev")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.0009, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train for (default: 5)")
    
    args = parser.parse_args()

    """
    TODO: Build your model Parameters. You can change the model architecture and hyperparameters as you wish.
            (e.g. change epochs, vocab_size, hidden_dim etc.)
    """
    # Model hyperparameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 200 # embedding dimension
    hidden_dim = 200  # hidden size of RNN
    num_layers = 3


    # Make Train Loader
    train_dataset = TextDataset(args.data_dir, 'train', args.vocab_size)
    
    # Save Train Vocab
    data = {}
    with open('train_vacab.p', 'wb') as f:
        data['sentences_vocab'] = train_dataset.sentences_vocab
        data['labels_vocab'] = train_dataset.labels_vocab
        pickle.dump(data, f)

    args.pad_idx = train_dataset.sentences_vocab.wtoi['<PAD>']
    train_loader, valid_loader = make_data_loader(train_dataset, args.batch_size, args.batch_first, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = model.to(device)
    print(train_dataset.labels_vocab.itow)
    # Training The Model
    train(args, train_loader, valid_loader, model)