"""
This code snippet helps extracting results: L2 distances SOTA and Adjacent Pixels from jsons of Deletion approach.
@author: J.BAIM
"""

import glob
import json
import numpy as np
import pandas as pd
import os


jsons = [
    "aurc_gbp", "aurc_ig", "aurc_sal", "aurc_grads", "aurc_smoothgrad", "aurc_gradcam"
]

for fold in jsons:
    #json_folder = f"/nobackup/mb282818/results/results300img/Deletion/{fold}"
    json_folder = f"./logs/logs/{fold}"
    if not os.path.exists(json_folder):
        print(f"Folder {json_folder} does not exist.")
        continue

    files = glob.glob(f"{json_folder}/*.json")

    df = pd.DataFrame()
    records = []
    
    idx_map = {20: 3, 40: 7, 50: 9}
    gt_index = 0

    for filepath in files:
        print(f"Processing file: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        l2_sota = np.array(data["L2_distances_md"])
        l2_rep = np.array(data["L2_distances_rp"])
        gt_identical = np.array(data["gt_identical"])
        inp_identical = np.array(data["inp_identical"])
        lrs_identical = np.array(data["lrs_identical"])
        sota_ae = np.array(data["ae_loss_masked"])
        rep_ae = np.array(data["ae_loss_rep"])

        row = {'img_name': data['img_name']}
        for p, idx in idx_map.items():
            val_sota = l2_sota[idx]
            val_rep = l2_rep[idx]
            val_rep_npix = inp_identical[idx][0]/(224*224)
            val_sota_npix = lrs_identical[idx][0]/(224*224)
            val_sota_ae = sota_ae[idx]
            val_rep_ae = rep_ae[idx]

            row[f'L2_{p}_sota'] = val_sota
            row[f'L2_{p}_rep'] = val_rep
            row[f'Ldist_{p}_rep'] = (val_rep_npix + val_rep_ae)/2
            row[f'Ldist_{p}_sota'] = (val_sota_npix + val_sota_ae)/2

        records.append(row)
    df = pd.DataFrame(records)
    # add a row with the mean values
    mean_row = df.mean(numeric_only=True).to_dict()
    mean_row['img_name'] = 'mean'
    df = df._append(mean_row, ignore_index=True)

    df.to_csv(f"./logs/logs/{fold}_l2_ldist.csv", index=False)
    print(df)

        