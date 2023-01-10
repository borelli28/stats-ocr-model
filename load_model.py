import pickle

# Load the trained model from the file
with open("./models/svm_model.pkl", "rb") as f:
    model = pickle.load(f)

# Make predictions on the new data using the model
# predictions = model.predict(X_test)
