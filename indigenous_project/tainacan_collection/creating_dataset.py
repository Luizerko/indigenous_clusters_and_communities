import glob
import json

from dataset_utils import text_normalization, date_conversion

import pandas as pd
import numpy as np

import requests
from tqdm import tqdm

# Getting data files coming from pagination
files = glob.glob('data/raw_data/*')

# Creating dataframe columns
with open(files[0]) as f:
    page_dict = json.load(f)

    # Explicit columns
    data_list = [[], [], [], [], []]
    data_columns = ['id', 'url', 'thumbnail', 'creation_date', 'modification_date']
    data_column_offset = len(data_list)

    # Iterating through 'data' to get non-explicit columns
    for key, value in page_dict['items'][0]['data'].items():    
        # Normalizing dataframe keys and values
        dataframe_key = text_normalization(value['label'], True)

        data_list.append([])
        data_columns.append(dataframe_key)

# Reading JSON and getting dataframe data
for file in files:
    with open(file) as f:
        # Loading JSON as a dictionary
        page_dict = json.load(f)

        for item in page_dict['items']:
            # Appending information coming from the explicit columns
            data_list[0].append(int(item['id']))
            data_list[1].append(item['url'])
            if item['thumbnail'] is False:
                data_list[2].append(None)
            else:
                data_list[2].append(item['thumbnail'][0])
            data_list[3].append(date_conversion(item['creation_date']))
            data_list[4].append(date_conversion(item['modification_date']))

            # Appending information coming from non-explicit columns
            for i, (key, value) in enumerate(item['data'].items()):
                # Treating special cases for date objects using indexes instead of comparing strings
                if i == 9 or i == 10 or i == 11:
                    try:
                        data_list[i+data_column_offset].append(date_conversion(value['value'], False))
                    except:
                        data_list[i+data_column_offset].append(value['value'])
                else:
                    data_list[i+data_column_offset].append(text_normalization(value['value'], False))

# Creating dataframe itself and adjusting data structure
data_list = np.array(data_list).transpose()
ind_df = pd.DataFrame(data_list, columns=data_columns)
ind_df = ind_df.set_index('id')
ind_df = ind_df.replace('', None)

# Printing dataframe information for sanity check and analysis
print(ind_df.info())

# Downloading the images
image_paths = []
for index, row in tqdm(ind_df.iterrows(), desc="Downloading Images", total=len(ind_df)):
    image_url = row['thumbnail']
    
    # When we dont have any images
    if type(image_url) is float:
        image_paths.append(None)

    # When we have images
    else:
        image_paths.append(f'data/images/{index}.jpg')

        try:
            response = requests.get(image_url)
            with open(image_paths[-1], 'wb') as f:
                f.write(response.content)
        except:
            print('Lost one image due to request')
            continue

# Updating dataframe with image information
ind_df['image_path'] = image_paths

# Saving data
ind_df.to_csv('data/indigenous_collection.csv')