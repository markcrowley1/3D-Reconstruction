
import sys
import pickle
import matplotlib.pyplot as plt

def main():
    # Get name of pickle file containing point cloud
    # Load and display saved point cloud as test
    filename = sys.argv[1]
    file = open(filename, "rb")
    data = pickle.load(file)
    loss = data["loss"]

    print(len(loss[60:]))
    plt.plot(range(60, 300), loss[60:])
    plt.xlabel('Epochs')
    plt.ylabel('Chamfer Distance')
    plt.title('Plot of Chamfer Distance vs Epochs')
    plt.show()
    

if __name__ == "__main__":
    main()