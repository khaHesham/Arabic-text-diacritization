from typing import List
from torch import nn
import torch
from dataset import DiacriticsDataset
from tqdm import tqdm


class BaseLineModel(nn.Module):
    def __init__(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        layers_units: List[int] = [256, 256, 256],
        use_batch_norm: bool = False,
    ):
        super(BaseLineModel,self).__init__()  
        self.targ_vocab_size = targ_vocab_size
        self.embedding = nn.Embedding(inp_vocab_size, embedding_dim)

        layers_units = [embedding_dim // 2] + layers_units

        layers = []

        for i in range(1, len(layers_units)):
            layers.append(
                nn.LSTM(
                    layers_units[i - 1] * 2,
                    layers_units[i],
                    bidirectional=True,
                    batch_first=True,
                )
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(layers_units[i] * 2))

        self.layers = nn.ModuleList(layers)
        self.projections = nn.Linear(layers_units[-1] * 2, targ_vocab_size)
        self.layers_units = layers_units
        self.use_batch_norm = use_batch_norm

    def forward(self, src: torch.Tensor, target=None):

        # for i in src:
        #     print(i)
        
        outputs = self.embedding(src) # outputs = [batch_size, sentence_len, embedding_dim]

        # embedded_inputs = [batch_size, src_len, embedding_dim]

        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.BatchNorm1d):
                outputs = layer(outputs.permute(0, 2, 1))
                outputs = outputs.permute(0, 2, 1)
                continue
            if i > 0:
                outputs, (hn, cn) = layer(outputs, (hn, cn))
            else:
                outputs, (hn, cn) = layer(outputs)

        predictions = self.projections(outputs)

        output = {"diacritics": predictions}

        return output
    
    
    
    def train(model,train_dataset, batch_size=512, epochs=10, learning_rate=0.01) :
        """
        Train the model on the training set.
        Args:
            train_dataset (torch.utils.data.Dataset): The training set.
            batch_size (int): The batch size.
            epochs (int): The number of epochs.
            learning_rate (float): The learning rate.
        """
        # Create a data loader from the training set
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # GPU configuration
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        if use_cuda:
            model = model.cuda()
            criterion = criterion.cuda()
        
        # Start training
        for epoch in range(epochs):
            running_loss = 0.0
            running_accuracy = 0.0
            for i, data in enumerate(tqdm(train_loader)):
                # Get the inputs
                inputs, labels = data[0], data[1]
                
                # print(inputs.shape)
                
                # (4) move the train input to the device
                '''
                move to gpu if available
                '''
                inputs = inputs.to(device)

                # (5) move the train label to the device
                labels = labels.to(device)
                
                # Forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs['diacritics'].view(-1, outputs['diacritics'].shape[-1]), labels.view(-1))
                
                running_loss += loss.item()
                accuracy = (torch.argmax(outputs['diacritics'],dim=-1) == labels).sum().item()
                running_accuracy += accuracy
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Backward pass
                loss.backward()
                
                # Optimize
                optimizer.step()
                
            # epoch loss
            epoch_loss = running_loss / len(train_dataset)

            # (13) calculate the accuracy
            epoch_acc = running_accuracy / (len(train_dataset) * train_dataset[0][0].shape[0])

            print(
                f'Epochs: {epoch + 1} | Train Loss: {epoch_loss} \
                | Train Accuracy: {epoch_acc}\n')
                    
                    
        print('Finished Training')
        
        
        
    def evaluate(model, test_dataset, batch_size=512):
        """
        Evaluate the model on the test set.
        Args:
            test_dataset (torch.utils.data.Dataset): The test set.
            batch_size (int): The batch size.
        Returns:
            float: The accuracy of the model on the test set.
        """
        # Create a data loader from the test set
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Set the model to evaluation mode
        model.eval()
        
        # Calculate accuracy on the test set
        with torch.no_grad():
            total = 0
            correct = 0
            for data in tqdm(test_loader):
                # Get the inputs
                inputs, labels = data[0], data[1]
                
                # Forward pass
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                acc = (torch.argmax(outputs['diacritics'],dim=-1) == test_label).sum().item()
                total_acc_test += acc
        total_acc_test /= (len(test_dataset) * test_dataset[0][0].shape[0])
        
        print(f'\nTest Accuracy: {total_acc_test}')
        return total_acc_test
        
        
        
        
        
def main():
    # # load the data
    train_dataset = DiacriticsDataset() 
    train_dataset.load('dataset/train.txt')

    # # creaate the model
    # model = BaseLineModel(
    #     inp_vocab_size=len(train_dataset.characters2id),
    #     targ_vocab_size=len(train_dataset.diacritic2id),
    # ) 
    
    
    
    
    model = BaseLineModel(
        inp_vocab_size=len(train_dataset.character_sentences[0]),
        targ_vocab_size=len(train_dataset.diacritic2id),
    ) 
    print(model)
    
    # train the model
    model.train(train_dataset)
    
    # evaluate the model
    test_dataset = DiacriticsDataset()
    test_dataset.load('dataset/val.txt')
    accuracy = model.evaluate(test_dataset)
    
    # print(f'Accuracy: {accuracy}')
    
    
if __name__ == '__main__':
    main()
    