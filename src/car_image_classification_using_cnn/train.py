from car_image_classification_using_cnn.model import Model
from car_image_classification_using_cnn.data import MyDataset

def train():
    dataset = MyDataset("data/raw")
    model = Model()
    # add rest of your training code here

if __name__ == "__main__":
    train()
