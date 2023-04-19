
import sys
import pickle
import matplotlib.pyplot as plt

def main():
    # Get name of pickle file containing point cloud
    # Load and display saved point cloud as test
    filename = sys.argv[1]
    file = open(filename, "rb")
    data = pickle.load(file)
    train_loss, val_loss = data["loss"], data["val_loss"]
    plt.plot(range(len(train_loss)), train_loss, label='Training Loss')
    plt.plot(range(len(train_loss)), val_loss, label='Validation Loss')
    plt.legend(loc="best")
    plt.xlabel('Epochs')
    plt.ylabel('Chamfer Distance')
    plt.title('Plot of Chamfer Distance vs Epochs')
    plt.show()
    

if __name__ == "__main__":
    main()