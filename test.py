import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from train import DanceDataset
from train import DanceGenreLSTM
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_dataset = DanceDataset("testing.pt")
        test_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

        num_genres = len(test_dataset.genre_to_index)
        input_size = test_dataset.X.shape[2]
        hidden_size = 256

        model = DanceGenreLSTM(input_size, hidden_size, num_genres)
        model.load_state_dict(torch.load("best_model.pth"))
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()

        test_loss, test_correct, test_total = 0, 0, 0

        pred_list, y_test = [], []

        with torch.no_grad():
                for X,y in test_loader:
                        X,y = X.to(device), y.to(device)
                        outputs = model(X)
                        loss = criterion(outputs, y)

                        test_loss += loss.item()
                        _, preds = outputs.max(1)
                        test_correct += preds.eq(y).sum().item()
                        test_total += y.size(0)

                        pred_list.extend(outputs.argmax(1).cpu().numpy())
                        y_test.extend(y.cpu().numpy())
        
        test_acc = 100.0 * (test_correct/test_total)
        test_loss_avg = test_loss/len(test_loader)

        print(f"Test Loss: {test_loss_avg:.4f}, Test Accuracy: {test_acc:.2f}%")

        print(classification_report(y_test, pred_list, target_names = list (test_dataset.genre_to_index.keys())))
        cm = confusion_matrix(y_test, pred_list, labels = list (test_dataset.genre_to_index.values()))
        plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list (test_dataset.genre_to_index.keys()))

        plot.plot()
        plt.savefig("test_results\\test_temporal_model.jpg")
        plt.show()

if __name__ == "__main__":
        main()


