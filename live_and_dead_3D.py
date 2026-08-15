import os
import numpy as np
import cv2
import zipfile
import pandas as pd

def dissect(name):
  part1 = name.split('/')
  time = part1[1].split(' ')[1]
  cell_line = part1[2].split('_')[0]
  cell_name = f'{time}_{cell_line}'
  return cell_name


zip_3d = zipfile.ZipFile(input("Insert the path of the .zip file: "))
imgs_3d=[img for img in zip_3d.namelist() if img.endswith('.tif')]

df = pd.DataFrame()
for k in range(int(len(imgs_3d)/2)):
  buf_live = zip_3d.read(imgs_3d[k*2])
  img_live = cv2.imdecode(np.frombuffer(buf_live, np.uint8), cv2.IMREAD_GRAYSCALE)
  ret_live, thresh_live=cv2.threshold(img_live, 25, 255, cv2.THRESH_BINARY)

  buf_dead = zip_3d.read(imgs_3d[k*2 + 1])
  img_dead = cv2.imdecode(np.frombuffer(buf_dead, np.uint8), cv2.IMREAD_GRAYSCALE)
  ret_dead, thresh_dead=cv2.threshold(img_dead, 80, 1, cv2.THRESH_BINARY)

  img_dead = img_dead*thresh_dead
  blue=np.zeros(img_live.shape).astype('uint8')
  merged= cv2.merge([blue, img_live, img_dead])
  merged_gray = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)

  dead_contours, dead_hierarchy =cv2.findContours(thresh_dead, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  ret_sphere, thresh_sphere=cv2.threshold(merged_gray, 25, 255, cv2.THRESH_BINARY)
  sphere_contours, sphere_hierarchy = cv2.findContours(thresh_sphere, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  total_dead_area = [cv2.contourArea(area) for area in dead_contours if cv2.contourArea(area) < 1000]
  dead_area_value = np.sum(total_dead_area)
  sphere_area = [cv2.contourArea(area) for area in sphere_contours]
  sphere_area_value = np.sum(sphere_area)


  #inside area
  live_contours, live_hierarchy = cv2.findContours(thresh_live, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  live_contours = [contour for contour in live_contours if cv2.contourArea(contour) > 0]
  live_area = [cv2.contourArea(contour)/0.04 for contour in live_contours]
  inside_area_list=[]
  for contour in live_contours:
    x, y, w, h=cv2.boundingRect(contour)
    window = thresh_dead[y:y+h, x:x+w]
    window_contours, window_hierarchy=cv2.findContours(window, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    window_contours = [pixel for pixel in window_contours if cv2.contourArea(pixel) > 0]
    inside_area = 0
    for pixel in window_contours:
      point=(int(pixel[0][0][0] + x), int(pixel[0][0][1]) + y)
      if cv2.pointPolygonTest(contour, point, measureDist=False) >= 0:
        inside_area = inside_area + cv2.contourArea(pixel)
    inside_area_list.append(inside_area)
  inside_area_list=[item/0.04 for item in inside_area_list]
  name = dissect(imgs_3d[k*2])


  row_data = [live_area, inside_area_list]
  col1 = pd.DataFrame(live_area, columns=[f'{name}_live_area'])
  col2 = pd.DataFrame(inside_area_list, columns=[f'{name}_dead_area'])
  row=pd.concat([col1, col2])
  df=pd.concat([df, row])

df = df.apply(lambda x: pd.Series(x.dropna().values))
df.to_csv('live_dead_3d_results_individuals.csv')
