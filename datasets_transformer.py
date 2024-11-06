import os
import tqdm
import pandas as pd

dir_datasets = "datasets\\"
datasets_name_arr = os.listdir(dir_datasets)
print(datasets_name_arr)
datasets_name_arr.remove("110-PT-BN-KP")
for lib_name in tqdm.tqdm(datasets_name_arr, desc="datasets", total=len(datasets_name_arr)):
    df = pd.DataFrame()
    desc = f"Processing {lib_name}"

    for doc_name in tqdm.tqdm(os.listdir(dir_datasets + lib_name +"\\docsutf8"), desc=desc, total=len(os.listdir(dir_datasets + lib_name +"\\docsutf8"))):
        with open(dir_datasets + lib_name +"\\docsutf8\\" + doc_name, 'r', encoding = 'utf-8') as file_text:
            with open(dir_datasets + lib_name +"\\keys\\" + doc_name.split('.')[0] + ".key", encoding = 'utf-8') as keys_file:
                k_arr = keys_file.readlines()
                if('\n' in k_arr):
                    k_arr.remove('\n')
                keys = []
                for k in k_arr:
                    keys.append(k.replace('\n', '').replace('\t', ''))
                df = df._append({"text": file_text.read(), "keys": keys}, ignore_index = True)

    df.to_csv("datasets_csv\\" + lib_name + ".scv")
    del df

