from imageai.Prediction.Custom import ModelTraining

model_trainer = ModelTraining()
model_trainer.setModelTypeAsResNet()
model_trainer.setDataDirectory("data")
model_trainer.trainModel(num_objects=2, num_experiments=2, enhance_data=True, batch_size=20, show_network_summary=True)
