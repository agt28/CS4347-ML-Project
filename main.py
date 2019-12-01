from cnnFireDetection.cnn_data_loader import  cnnDataLoader
from cnnFireDetection.cnn_training_model import cnnFireDetectionModel

if __name__== "__main__":
  data = cnnDataLoader()
  model = cnnFireDetectionModel()
  dataSet = data.create_training_data()
  data.storeData(dataSet)

  # Train Model
  X, y = data.loadData()
  model.runModelOpt(X,y)


