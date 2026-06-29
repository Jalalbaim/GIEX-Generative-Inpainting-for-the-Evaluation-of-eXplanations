"""
ce script permet de compter le nombre de pixels adjacents identiques dans une image.
Il compte les pixels adjacents identiques horizontalement, verticalement et en diagonale.
@author: J.BAIM
"""
import numpy as np
from PIL import Image
import pandas as pd

def adjacent_identical_pixels_rgb(image_path):
    img = np.array(Image.open(image_path), dtype=np.int32)

    total = 0
    for c in range(3):
        ch = img[:, :, c]

        # horizontal / vertical
        diff_h = ch[:, 1:] - ch[:, :-1]
        diff_v = ch[1:, :] - ch[:-1, :]

        # diagonales
        diff_rd = ch[1:, 1:] - ch[:-1, :-1]
        diff_ru = ch[:-1, 1:] - ch[1:, :-1]

        total += (diff_h == 0).sum()
        total += (diff_v == 0).sum()
        total += (diff_rd == 0).sum()
        total += (diff_ru == 0).sum()

    return int(total)


def count_identical_adjacent(img):

    img = np.array(img, dtype=np.int32)

    # Droite et bas
    same_right = (img[:, :-1]   == img[:, 1:]).all(axis=2)
    same_down  = (img[:-1]      == img[1:]).all(axis=2)

    # Diagonales
    same_diag_rd = (img[:-1, :-1] == img[1:, 1:]).all(axis=2)   # bas-droite
    same_diag_ru = (img[1:, :-1]  == img[:-1, 1:]).all(axis=2)  # haut-droite

    right_equals = int(same_right.sum())
    down_equals  = int(same_down.sum())
    diag_rd_eq   = int(same_diag_rd.sum())
    diag_ru_eq   = int(same_diag_ru.sum())

    total = right_equals + down_equals + diag_rd_eq + diag_ru_eq
    return total, right_equals, down_equals, diag_rd_eq, diag_ru_eq

if __name__ == '__main__':

    # methods = ["gradcam", "gbp", "smoothgrad", "grads", "ig", "sal"]

    # for method in methods:
    #     print(f"Processing method: {method}")

    #     folder = f"/nobackup/mb282818/repaint_res_normal/imagenette_sub2_{method}"
        
    #     subfolders = ["gt", "gt_masked", "inpainted"]
    #     df = {}

    #     for subfolder in subfolders:
    #         list_img = glob.glob(f"{folder}/{subfolder}/*.png")
    #         df[subfolder] = []
    #         for img_path in list_img:
    #             count = count_identical_adjacent(img_path)
    #             df[subfolder].append({
    #                 'img_name': img_path.split("/")[-1],
    #                 'count': count
    #             })
        
    #     # merge dataframes on img_name
    #     df_final = pd.DataFrame()
    #     for subfolder in subfolders:
    #         temp_df = pd.DataFrame(df[subfolder])
    #         temp_df.rename(columns={'count': f'count_{subfolder}'}, inplace=True)
    #         if df_final.empty:
    #             df_final = temp_df
    #         else:
    #             df_final = df_final.merge(temp_df, on='img_name', how='outer')


    #     df_final.to_csv(f"adjacent_identical_{method}.csv", index=False)
    #     print(f"Results saved to adjacent_identical_{method}.csv")

    # methods = ["gradcam", "gbp", "smoothgrad", "grads", "ig", "sal"]

    # masked = []
    # inpainted = []

    # for method in methods:
    #     print(f"Processing method: {method}")

    #     csv_path = f"adjacent_identical_{method}.csv"

    #     try:
    #         df = pd.read_csv(csv_path)
    #     except FileNotFoundError:
    #         print(f"CSV file not found: {csv_path}")
    #         continue

    #     img = df["img_name"].iloc[0]
    #     gt = df["count_gt"].iloc[0]
    #     gt_masked = df["count_gt_masked"].iloc[0]
    #     inpainted = df["count_inpainted"].iloc[0]

    #     print(f"Image: {img}, GT: {gt}, GT Masked: {gt_masked}, Inpainted: {inpainted}")
        
    #     gt_total = int(gt.split(',')[0][1:])
    #     gt_masked_total = int(gt_masked.split(',')[0][1:])
    #     inpainted_total = int(inpainted.split(',')[0][1:])

    #     print(f"Total identical adjacent pixels - GT: {gt_total}, GT Masked: {gt_masked_total}, Inpainted: {inpainted_total}")

    img_gt = "./logs/water_ouzel_iterated_ig/gt/water_ouzel_img.png"
    img_gt_masked = "./logs/water_ouzel_iterated_ig/gt_masked/water_ouzel_img.png"
    img_inpainted = "./logs/water_ouzel_iterated_ig/inpainted/water_ouzel_img.png"

    img_gradcam_masked = "./logs/water_ouzel_iterated grad-cam/gt_masked/water_ouzel_img.png"
    img_gradcam_inpainted = "./logs/water_ouzel_aurc_gradcam/water_ouzel_img_pct25.png"

    # Count identical adjacent pixels
    count_gt, right_equals, down_equals, diag_rd_eq, diag_ru_eq = count_identical_adjacent(Image.open(img_gt))
    count_gt_masked, right_equals_masked, down_equals_masked, diag_rd_eq_masked, diag_ru_eq_masked = count_identical_adjacent(Image.open(img_gt_masked))
    count_inpainted, right_equals_inpainted, down_equals_inpainted, diag_rd_eq_inpainted, diag_ru_eq_inpainted = count_identical_adjacent(Image.open(img_inpainted))

    count_gradcam_masked, right_equals_gradcam_masked, down_equals_gradcam_masked, diag_rd_eq_gradcam_masked, diag_ru_eq_gradcam_masked = count_identical_adjacent(Image.open(img_gradcam_masked))
    count_gradcam_inpainted, right_equals_gradcam_inpainted, down_equals_gradcam_inpainted, diag_rd_eq_gradcam_inpainted, diag_ru_eq_gradcam_inpainted = count_identical_adjacent(Image.open(img_gradcam_inpainted))


    print(f"gt: ({count_gt}, {right_equals}, {down_equals}, {diag_rd_eq}, {diag_ru_eq})")
    print(f"gt_masked: ({count_gt_masked}, {right_equals_masked}, {down_equals_masked}, {diag_rd_eq_masked}, {diag_ru_eq_masked})")
    print(f"inpainted: ({count_inpainted}, {right_equals_inpainted}, {down_equals_inpainted}, {diag_rd_eq_inpainted}, {diag_ru_eq_inpainted})")
    print(f"gradcam_masked: ({count_gradcam_masked}, {right_equals_gradcam_masked}, {down_equals_gradcam_masked}, {diag_rd_eq_gradcam_masked}, {diag_ru_eq_gradcam_masked})")
    print(f"gradcam_inpainted: ({count_gradcam_inpainted}, {right_equals_gradcam_inpainted}, {down_equals_gradcam_inpainted}, {diag_rd_eq_gradcam_inpainted}, {diag_ru_eq_gradcam_inpainted})")


