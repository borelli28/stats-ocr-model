# stats-ocr-model

This will be a OCR(Optical Character Recognition) ML model to extract statistics from the mlb.com website. It will work by taking screenshots of each player stats page and then feeding those images to the model in order to extract the statistics of each player into a CSV file.

### Usage

Creates **players.csv** with All players: **Names**, **ID's** & stats **URL's**
```bash
python3 get_players.py https://www.mlb.com/players
```

Create the images(screenshot) of all players stats. Reads from the **players.csv**file. Images are stored in **stats-ocr-model/assets/images** folder
```bash
python3 get_images.py
```


### License

[MIT](https://choosealicense.com/licenses/mit/)