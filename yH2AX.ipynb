import pandas as pd
import zipfile
import os
import cv2
import numpy as np

def get_channel(list_img):
  try:
    img=cv2.imread(list_img[0], 0)
  except IndexError:
    img=np.zeros((1960, 1960), dtype=np.uint8)
  return img

def rescale(img):
  img = img - np.min(img)
  img = (img/np.max(img)) * 255
  img = np.floor(img)
  return img.astype(np.uint8)

def watershed(fg_img, bg_img, color_img):
  unknown = cv2.subtract(bg_img, fg_img)
  ret, markers = cv2.connectedComponents(fg_img)
  markers = markers+1
  markers[unknown==255] = 0
  markers = cv2.watershed(color_img, markers)
  final_mask = np.zeros(fg_img.shape, np.uint8)
  final_mask[markers == -1] = 255
  return final_mask

def get_name(string):
  items=string.split('/')
  return items

def get_group(string):
  if "1 " in string:
    group="DMSO"
  elif "2 " in string:
    group="Zika"
  elif "3 " in string:
    group="IMSO"
  elif "4 " in string:
    group="Zika+IMSO"
  return group



zip_path=input("Insert the location of the .zip file: ")

df=pd.DataFrame()
columns=['img_class', 'red', 'group', 'positive_cells', 'positive_red_value','nuclei_count', 'nuclei_area', 'overall_red']

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
  file_names=zip_ref.namelist()
  filtered_folders=[x for x in file_names if '.tif' not in x]
  sub_folders=[x for x in filtered_folders if 'Phase' in x]
  sub_folders=sorted(sub_folders)

  for folder in sub_folders:
    img_files_zip=[x for x in file_names if ('.tif' in x) and (folder in x)]
    [zip_ref.extract(x, '/content') for x in img_files_zip]
    dapi_name = [f'/content/{x}' for x in img_files_zip if 'DAPI' in x]
    cy_name = [f'/content/{x}' for x in img_files_zip if 'CY5' in x]

    name=get_name(folder)[-2]
    group=get_group(folder)

    try:
      cy_img=rescale(cv2.imread(cy_name[0], 0))
      gfp_img= rescale(get_channel(gfp_name))
      dapi_img= rescale(get_channel(dapi_name))

      merge_img= cv2.merge((dapi_img, gfp_img, cy_img))

      _, dapi_thresh=cv2.threshold(dapi_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

      dist_gfp = cv2.distanceTransform(dapi_thresh, cv2.DIST_L2, 5).astype(np.uint8)
      _, mask_fg = cv2.threshold(dist_gfp, 10, 255, cv2.THRESH_BINARY)
      dapi_watershed=watershed(mask_fg, dapi_thresh, merge_img)

      cnt_dapi, hierarchy=cv2.findContours(dapi_watershed[1:-1, 1:-1], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      filter_dapi=[cnt for cnt in cnt_dapi if cv2.contourArea(cnt) > 500]

      nuclei_mask=np.zeros(dapi_img.shape, dtype=np.uint8)
      cv2.drawContours(nuclei_mask, filter_dapi, -1, 1, -1)

      _, cy5_thresh=cv2.threshold(cy_img, 100, 255, cv2.THRESH_BINARY)
      cy5_thresh=nuclei_mask * cy5_thresh

      cy5_count=cv2.mean(cy5_thresh)[0] * cy_img.shape[0] * cy_img.shape[1]/255
      cy5_mean_count=cy5_count/len(filter_dapi)

      positive_cells=0
      positive_cnts=[]
      dapi_areas=[cv2.contourArea(cnt) for cnt in filter_dapi]
      avg_dapi=sum(dapi_areas)/len(dapi_areas)
      for cnt in filter_dapi:
        temp_mask=np.zeros(cy_img.shape, dtype=np.uint8)
        cv2.drawContours(temp_mask, [cnt], -1, 255, -1)
        if cv2.mean(cy5_thresh, mask=temp_mask)[0] > 0:
          positive_cells=positive_cells + 1
          positive_cnts.append(cnt)

      try:
        cy5_mean_positives=cy5_count/len(positive_cnts)
      except ZeroDivisionError:
        cy5_mean_positives=0

      row_df={'img_class': name, 'red': cy5_mean_count, 'group': group, 'positive_cells': positive_cells,\
              'positive_red_value': cy5_mean_positives, 'nuclei_count':len(filter_dapi),\
              'nuclei_area': avg_dapi, 'overall_red': cv2.mean(cy_img)[0]}
      row=pd.DataFrame([row_df], columns=columns)
      df=pd.concat([df,row])

    except IndexError:
      pass
