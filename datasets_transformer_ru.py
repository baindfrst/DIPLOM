import os
import tqdm
import pandas as pd
import jsonlines

dir_datasets = "ru_datasets\\"
name_out = {"cyberleninka": 5, "habrahabr": 4, "ng": 2, "russia_today": 8}
for key in name_out.keys():
    df = pd.DataFrame()
    for num in range(name_out[key]):
        with jsonlines.open(dir_datasets + key + "_" + str(num) + ".jsonlines" +"\\" + key + "_" + str(num) + ".jsonlines", 'r') as reader:
            for line in tqdm.tqdm(reader, desc = key + "_" + str(num)):
                df = df._append({"text": line['content'], "keys": line['keywords']}, ignore_index = True)

    df.to_csv("datasets_csv\\" + key + ".scv")
    del df

