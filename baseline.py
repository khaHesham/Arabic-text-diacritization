from typing import List
from torch import nn
import torch


class BaseLineModel(nn.Module):
    def __init__(
        self,
        inp_vocab_size: int,
        targ_vocab_size: int,
        embedding_dim: int = 512,
        layers_units: List[int] = [256, 256, 256],
        use_batch_norm: bool = False,
    ):
        super().__init__()  
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

    def forward(self, src: torch.Tensor, lengths: torch.Tensor, target=None):

        outputs = self.embedding(src)

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
    
    
    
    def train(model,train_dataset, batch_size=512, epochs=5, learning_rate=0.01) :
        """
        Train the model on the training set.
        Args:
            train_dataset (torch.utils.data.Dataset): The training set.
            batch_size (int): The batch size.
            epochs (int): The number of epochs.
            learning_rate (float): The learning rate.
        """
        # Create a data loader from the training set
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Define the optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Start training
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader):
                # Get the inputs
                inputs, labels = data['sentence'], data['diacritics']
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Optimize
                optimizer.step()
                
                # Print statistics
                running_loss += loss.item()
                if i % 100 == 99:    # Print every 100 mini-batches
                    print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100}')
                    running_loss = 0.0
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
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        # Define the loss function
        criterion = nn.CrossEntropyLoss()
        
        # Set the model to evaluation mode
        model.eval()
        
        # Calculate accuracy on the test set
        with torch.no_grad():
            total = 0
            correct = 0
            for data in test_loader:
                # Get the inputs
                inputs, labels = data['sentence'], data['diacritics']
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # Set the model back to training mode
        model.train()
        
        return correct / total
        
        
        
        
        
def main():
    # load the data
    train_dataset = DiacriticsDataset('data/train.csv') 
    
    # Train the model
    model = BaseLineModel(len(train_dataset.vocab), len(train_dataset.diacritics))
    model.train(train_dataset)
    
    # Evaluate the model
    test_dataset = DiacriticsDataset('data/test.csv')
    accuracy = model.evaluate(test_dataset)
    
    print(f'Accuracy: {accuracy}')
    
    
    
if __name__ == '__main__':
    main()
    