import sys
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import csv


# Create a ChromeOptions object and set the headless parameter to True
options = webdriver.ChromeOptions()
options.headless = True

# Set the path to the webdriver
driver = webdriver.Chrome(options=options, executable_path="/chromedriver_mac64/chromedriver")

# Open the file for reading
with open('players.csv', 'r') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile)

    # Skip the first row (column names)
    next(reader)

    # Read the rows of the file
    for row in reader:
        # Get the url
        url = str(row[2])
        print(url)

        # Load the web page
        driver.get(url)

		# IMPORTANT: Wait 3 second between each URL request to not overwhelm the server
        sleep(3)

		# Define a lambda function that returns the value of a JavaScript expression
        S = lambda X: driver.execute_script("return document.body.parentNode.scroll"+X)

		# Set the size of the browser window to the dimensions of the webpage
        driver.set_window_size(S("Width"), S("Height"))

		# Take and save screenshot
        name = str(row[0])
        driver.find_element(By.TAG_NAME, "body").screenshot(f"./assets/images/{name}.png".format(name))

driver.quit()

print("Done")