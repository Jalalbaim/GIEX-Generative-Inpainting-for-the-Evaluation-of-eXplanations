"""
Ce code permet de recuperer un csv avec les valeurs de AURC et AUDC pour chaque méthode.
les outputs sont sur RePaint/logs/metrics/auic.csv or aurc.csv
@author: J.BAIM
"""

import glob
import json
import numpy as np
import pandas as pd
import os

methods = ["gbp", "sal", "smoothgrad", "grads", "ig", "gradcam", "lrp"]
seeds = ["seed42"]

for s in seeds:
    df = {}
    
    for m in methods:
        json_folder = f'./results/aurc_{m}_miss/{s}'
        files = glob.glob(f'{json_folder}/*.json')

        imgs = []
        aurcs = []
        audcs = []

        for filepath in files:
            print(f"Processing file: {filepath}")
            aurc = 0.0
            audc = 0.0

            with open(filepath, 'r') as f:
                data = json.load(f)

            prob_inp = data['probabilities_inp']
            prob_masked = data['probabilities_masked']
            pixels = data['pixels_removed']
            #pixels = data['pixels_left']

            pixels.insert(0, 0.00)
            #pixels.append(1) 
            x = np.array(pixels) * 100

            y_rep = np.array(prob_inp)
            y_del = np.array(prob_masked)

            aurc = np.trapz(y_rep, x)
            audc = np.trapz(y_del, x)

            print(f"AURC: {aurc}")
            print(f"AUDC: {audc}")

            imgs.append(data['img_name'])
            aurcs.append(aurc)
            audcs.append(audc)
        
        df[m] = {
            'img_name': imgs,
            'aurc_r': aurcs,
            'aurc_masked': audcs
        }

    df_final = pd.DataFrame()
    for m in methods:
        temp_df = pd.DataFrame(df[m])
        temp_df.rename(columns={'aurc_r': f'aurc_r_{m}', 'aurc_masked': f'aurc_masked_{m}'}, inplace=True)
        if df_final.empty:
            df_final = temp_df
        else:
            df_final = df_final.merge(temp_df, on='img_name', how='outer', suffixes=('_final', '_temp'))

    # compute means for numeric columns and append as a final row with img_name='mean'
    mean_row = df_final.select_dtypes(include=[np.number]).mean()
    mean_row = mean_row.to_dict()
    mean_row['img_name'] = 'mean'
    df_final = pd.concat([df_final, pd.DataFrame([mean_row])], ignore_index=True)
    print(f"Seed {s}: {len(df_final)} rows")

    # Save one CSV per seed
    output_path = f'./results/aurc_{s}_lrp_included.csv'
    df_final.to_csv(output_path, index=False)
    print(f"DataFrame saved to {output_path}\n")