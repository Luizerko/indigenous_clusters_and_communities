import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

# Setup Selenium WebDriver
driver = webdriver.Chrome(executable_path='/chromedriver-linux64/chromedriver')
driver.get("https://graffiti-database.com/")

# Function for scrolling down and loading more images
def load_all_images(driver, scroll_pause_time=2):
    last_height = driver.execute_script("return document.body.scrollHeight")
    
    while True:
        # Scrolling down to the bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Wait for new images to load
        time.sleep(scroll_pause_time)

        # Compute new scroll height and compare with last one
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

# Scrolling to load all images
load_all_images(driver)

# Selecting all the image elements on page
image_elements = driver.find_elements(By.CSS_SELECTOR, "div.masonry-grid-item.absolute a")

# Creating a directory to save the images
output_dir = "graffiti_database"

# Looping through images and downloading
for index, img_element in enumerate(image_elements):

    img_url = img_element.get_attribute("href")
    if '/piece/' not in img_url:
        continue
    img_url = 'https://graffiti-database.com/' + img_url

    print(f"Downloading image {index + 1}: {img_url}")
    img_data = requests.get(img_url).content
    img_filename = os.path.join(output_dir, f"image_{index + 1}.jpg")
    with open(img_filename, "wb") as img_file:
        img_file.write(img_data)

# Close the WebDriver
driver.quit()

print("Image scraping completed.")