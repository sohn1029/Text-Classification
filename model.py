from unicodedata import bidirectional
import torch
import torch.nn as nn
from torch.nn import functional as F

class BaseModel(nn.Module):
    '''
    input_size -> text vocab size
    '''
    def __init__(self, input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first):
        super(BaseModel, self).__init__()

        self.num_layers = num_layers
        self.batch_first = batch_first   
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.bi = True
        """
        TODO: Implement your own model. You can change the model architecture.
        """
        self.embedding = nn.Embedding(input_size, embedding_dim)

        self.lstm_bi = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=batch_first, bidirectional=self.bi)

        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.bn = nn.BatchNorm1d(self.hidden_dim)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(self.hidden_dim, output_size)


    # the size of x in forward is (seq_length, batch_size) if batch_first=False
    def forward(self, x):
        batch_size = x.size(0) if self.batch_first else x.size(1)

        #h_0: (num_layers * num_directions, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim)

        embedding = self.embedding(x)
        outputs2, hidden2 = self.lstm_bi(embedding, None)
        #32 64 64
        #4 64 64
        if self.bi:
            outputs2 = outputs2[:,:,:self.hidden_dim] * outputs2[:,:,self.hidden_dim:self.hidden_dim*2]
        outputs2 = outputs2.permute(1,0,2)

        result = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.num_layers):
            
            attn_weights = torch.bmm(outputs2, hidden2[i].squeeze(0).unsqueeze(2)).squeeze(2)
   
            soft_attn_weights = F.softmax(attn_weights, 1)
 
            new_hidden = torch.bmm(outputs2.transpose(1,2), soft_attn_weights.unsqueeze(2)).squeeze(2)
            
            result += new_hidden
            #result += outputs2
        #64 128
        result /= self.num_layers
        result = self.fc(result)
        result = self.relu(self.bn(result))
        result = self.fc2(result)

        return result, hidden2, soft_attn_weights

