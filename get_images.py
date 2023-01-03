import sys
from time import sleep
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By


options = webdriver.ChromeOptions()
options.headless = True

# Set the path to the webdriver
driver = webdriver.Chrome(options=options, executable_path="/chromedriver_mac64/chromedriver")

# Get URL parameter passed when file is executed
arg = sys.argv[1]
print(f"Parameter passed {arg}".format(arg))

# Load the web page
driver.get(f"https://www.{arg}".format(arg))

sleep(1)

S = lambda X: driver.execute_script("return document.body.parentNode.scroll"+X)
driver.set_window_size(S("Width"),S("Height")) # May need manual adjustment                                                                                                               

driver.find_element(By.TAG_NAME, "body").screenshot("./assets/images/screenshot.png")

driver.quit()

print("Done")