"""
This script computes various metrics (drop, increase, gain) based on probabilities
from input and masked images. It processes JSON files containing these probabilities,
calculates the metrics, and saves the results in a CSV file. The script is designed
to handle multiple methods and outputs the results for each method in a structured format.

@author: J.BAIM

"""
import glob
import json
import numpy as np
import pandas as pd

def drop(val_gt, val_res):
    return max(0, (val_gt - val_res)) / val_gt


def increase(val_gt, val_res):
    return 1 if val_res > val_gt else 0


def gain(val_gt, val_res):
    den = 1 - val_gt
    return max(0, (val_res - val_gt)) / den if den > 0 else 0

jsons = [
    "aurc_gbp_miss", "aurc_ig_miss", "aurc_sal_miss", "aurc_grads_miss", "aurc_smoothgrad_miss", "aurc_gradcam_miss", "aurc_lrp_miss"
]

for fold in jsons:
    json_folder = f"./results/{fold}/seed42"
    mode = fold.split('_')[0]  # 'aurc' ou 'auic'
    files = glob.glob(f"{json_folder}/*.json")

    df = pd.DataFrame()
    records = []

    if mode == 'aurc':
        idx_map = {10: 2, 20: 4, 30: 6, 40: 8}
        gt_index = 0
    else:
        idx_map = {10: 0, 20: 1, 30: 2, 40: 3}
        gt_index = -1

    for filepath in files:
        print(f"length of file: {len(files)}")
        with open(filepath, 'r') as f:
            data = json.load(f)

        probs_inp = np.array(data['probabilities_inp'])
        probs_mask = np.array(data['probabilities_masked'])
        val_gt = probs_inp[gt_index]

        row = { 'img_name': data['img_name'] }
        for p, idx in idx_map.items():
            val_inp = probs_inp[idx]
            val_mask = probs_mask[idx]
            row[f'val_{p}_inp'] = val_inp
            row[f'val_{p}_masked'] = val_mask
            # Drop
            row[f'val_{p}_drop'] = drop(val_gt, val_inp)
            row[f'val_{p}_drop_masked'] = drop(val_gt, val_mask)
            # Increase
            row[f'val_{p}_inc'] = increase(val_gt, val_inp)
            row[f'val_{p}_inc_masked'] = increase(val_gt, val_mask)
            # Gain
            row[f'val_{p}_gain'] = gain(val_gt, val_inp)
            row[f'val_{p}_gain_masked'] = gain(val_gt, val_mask)

        records.append(row)

    df = pd.DataFrame.from_records(records)

    print(f"=== Results pour {fold} ({mode}) ===")
    for metric in ['drop', 'inc', 'gain']:
        print(f"-- {metric.upper()} --")
        for p in [10, 20, 30, 40]:
            col = f'val_{p}_{metric}'
            col_m = f'val_{p}_{metric}_masked'
            if metric == 'inc':
                m = df[col].mean()*100
                m_m = df[col_m].mean()*100
                print(f"Increase @{p}% inp : {round(m,2)}%")
                print(f"Increase @{p}% masked : {round(m_m,2)}%")
            else:
                m = df[col].mean() * 100
                m_m = df[col_m].mean() * 100
                print(f"Average {metric} @{p}% inp : {round(m,2)}%")
                print(f"Average {metric} @{p}% masked : {round(m_m,2)}%")
    print()

    out_csv = f"./results/{fold}_values.csv"
    df.to_csv(out_csv, index=False)

    out_txt = f"./results/{fold}_metrics.txt"
    with open(out_txt, 'w') as f:
        for metric in ['drop', 'inc', 'gain']:
            label = ''
            if mode == 'aurc' and metric == 'drop':
                label = 'higher is better'
            elif mode == 'aurc' and metric in ['inc', 'gain']:
                label = 'lower is better'
            elif mode == 'auic' and metric == 'drop':
                label = 'lower is better'
            # elif mode == 'auic' and metric in ['inc', 'gain']:
            else:
                label = 'higher is better'
            f.write(f"Metric {metric.upper()} ({label})\n")
            for p in [10, 20, 30, 40]:
                col = f'val_{p}_{metric}'
                col_m = f'val_{p}_{metric}_masked'
                if metric == 'inc':
                    m = df[col].mean()*100
                    m_m = df[col_m].mean()*100
                    f.write(f"@{p}% inp: {round(m,2)}%\n")
                    f.write(f"@{p}% masked: {round(m_m,2)}%\n")
                else:
                    m = df[col].mean() * 100
                    m_m = df[col_m].mean() * 100
                    f.write(f"@{p}% inp: {round(m,2)}%\n")
                    f.write(f"@{p}% masked: {round(m_m,2)}%\n")
