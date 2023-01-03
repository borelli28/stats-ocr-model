import sys
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import re
import csv


# Create a ChromeOptions object and set the headless parameter to True
options = webdriver.ChromeOptions()
options.headless = True

# Set the path to the webdriver
driver = webdriver.Chrome(options=options, executable_path="/chromedriver_mac64/chromedriver")

# Get URL parameter passed when file is executed
arg = sys.argv[1]
print(f"Parameter passed: {arg}".format(arg))

# Load the web page
driver.get(arg)

sleep(1)

# Define a lambda function that returns the value of a JavaScript expression
S = lambda X: driver.execute_script("return document.body.parentNode.scroll"+X)
# Set the size of the browser window to the dimensions of the webpage
driver.set_window_size(S("Width"),S("Height"))

# Extract href values of all players <a> in the web page
hrefs = []
elements = driver.find_elements(By.CLASS_NAME, "p-related-links__link")
for i in elements:
	element = i.get_attribute("href")
	print(f"element: {element}".format(element))
	hrefs.append(element)

data = []
# Strip strings into Name, ID & URL
for i in hrefs:
	# Original string
	s = i

	# Extract the player name and ID using a regular expression
	m = re.search(r"/player/(.+)-(\d+)", s)
	name = m.group(1)
	id = m.group(2)

	print(f"Name: {name}")  # Output: Name: tory-abbott
	print(f"ID: {id}")  # Output: ID: 676265

	obj = [name, id, i]
	data.append(obj)

# Write data to CSV
with open("players.csv", "w", newline="") as csvfile:
  # Create a CSV writer object
  writer = csv.writer(csvfile)
  
  # Write the header row
  writer.writerow(["Name", "Player ID", "URL"])
  
  for i in data:
	  # Write the player data
	  writer.writerow([i[0], i[1], i[2]])

driver.quit()

print("Done")