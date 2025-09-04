import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

class DanceDataset(Dataset):
    def __init__(self, pt_file):
        data = torch.load(pt_file)
        self.X = data["data"]
        self.y = data["labels"]
        self.genre_to_index = data["genre_to_index"]

        #flatten to single vector per frame
        # N videos for training, T frames (all 300), J joints (17), C = 3 coordinates (3d)
        N, T, J, C = self.X.shape
        self.X = self.X.view(N, T, J*C)
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class DanceGenreLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_genres, num_layers = 2, dropout = 0.3):
        super(DanceGenreLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )

        self.fully_connected = nn.Linear(hidden_size, num_genres)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out.mean(dim = 1)
        out = self.fully_connected(out)
        return out
    
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = DanceDataset("training.pt")
    train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)

    num_genres = len(train_dataset.genre_to_index)
    input_size = train_dataset.X.shape[2]
    hidden_size = 256

    model = DanceGenreLSTM(input_size, hidden_size, num_genres)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(20):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(y).sum().item()
            total += y.size(0)
        
        acc = 100.0 * (correct/total)
        print(f"Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.2f}%")
    
if __name__ == "__main__":
    main()