# stats-ocr-model

This will be a OCR(Optical Character Recognition) ML model to extract statistics from the mlb.com website. It will work by taking screenshots of each player stats page and then feeding those images to the model in order to extract the statistics of each player into a CSV file.

## Model accuracy

The accuracy of the model on the test data is **93.1%**.

## Usage

Creates **players.csv** with All players: **Names**, **ID's** & stats **URL's**
```bash
python3 get_players.py https://www.mlb.com/players
```


Create the images(screenshot) of all players stats. Reads from the **players.csv** file. Images are stored in **stats-ocr-model/assets/images** folder
```bash
python3 get_images.py
```


Convert images to grayscale and crops them for better readability by OCR model
```bash
python3 pre_process.py
```

## ocr_model.py Explained
"This code appears to be implementing an OCR model using the SVM (support vector machine) algorithm. The model is trained to recognize text in images by extracting features from the images and labels from the annotations in a training dataset, then using these to train an SVM model. The model can then be used to make predictions on new images. The features of the images are extracted by cropping and resizing the images, then flattening the resulting image arrays and the labels are extracted from the names of the objects in the annotations. The model is evaluated by comparing the predictions made by the model to the ground truth labels and calculating the accuracy as a percentage." - ChatGPT

## License

[MIT](https://choosealicense.com/licenses/mit/)