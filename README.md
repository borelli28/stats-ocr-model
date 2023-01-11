# stats-ocr-model

This will be a OCR(Optical Character Recognition) ML model to extract statistics from the mlb.com website. It will work by taking screenshots of each player stats page and then feeding those images to the model in order to extract the statistics of each player into a CSV file.

## Model accuracy

Current average model accuracy in test data: **60.0%**

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


Trains SVM OCR model with annotations and labeled images in ./assets directory
```bash
python3 svm_ocr_model.py
```

## Training the model with your data

Annotations file format: `Pascal VOC XML`

Annotations should include categories and labels

*Categories*:
- Numbers: `number`
- Words & Letters: `word`
- Characters like this; `-`, `.`, `/`: `symbol`

*Labels*:
- Values(`HR`, `2`, `Season`, etc.)

Annotations are saved in `stats-ocr-model/assets/annotations` directory

Labeled images are saved in `stats-ocr-model/assets/labeled-images` directory

Image labeling software: https://github.com/NaturalIntelligence/imglab
Installation guide: https://github.com/NaturalIntelligence/imglab/blob/master/docs/guide.md/#offline-installation

## License

[MIT](https://choosealicense.com/licenses/mit/)