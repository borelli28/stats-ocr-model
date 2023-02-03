# stats-ocr-model

This will be a OCR(Optical Character Recognition) model to extract statistics from the mlb.com website. It will work by taking screenshots of each player stats page and then feeding those images to the model in order to extract the statistics of each player into a CSV file.

## Model accuracy

- SVM model: Current average accuracy in test data: **75.0%**

- CNN model: Current average accuracy in test data: **10.0%**

## Install Linux/MacOS

Clone repo
```git clone https://github.com/borelli28/stats-ocr-model.git```

Create virtual env for packages
```python3 -m venv myenv```

Activate virtual env
```source myenv/bin/activate```

Move into repo
```cd stats-ocr-model```

Install packages in requirements.txt
```pip install -r requirements.txt```

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


Trains CNN OCR model with annotations and labeled images in ./assets directory
```bash
python3 cnn_ocr_model.py
```

## Training the model with your data

Annotations file format: `Pascal VOC XML`

Annotations should include the category(`characters`) and the label value(`MLB`, `1`, `.`, etc.)

Annotations are saved in `stats-ocr-model/assets/annotations` directory

Labeled images are saved in `stats-ocr-model/assets/labeled-images` directory

Image labeling software: https://github.com/NaturalIntelligence/imglab
Installation guide: https://github.com/NaturalIntelligence/imglab/blob/master/docs/guide.md/#offline-installation

## License

[MIT](https://choosealicense.com/licenses/mit/)
