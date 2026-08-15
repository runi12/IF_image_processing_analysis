import os
import numpy as np
import zipfile
import pandas as pd

def fft_padding(img, kernel):
  p = img.shape[0] + kernel.shape[0] - 1
  q = img.shape[1] + kernel.shape[1] - 1
  new_img = np.zeros((p, q))
  new_img[0:img.shape[0], 0:img.shape[1]]=img
  new_kernel = np.zeros((p, q))
  new_kernel[0:kernel.shape[0], 0:kernel.shape[1]]=kernel
  return new_img, new_kernel

def convolve(img, kernel, iter):
  new_img, new_kernel = fft_padding(img, kernel)
  fft_img=np.fft.fft2(new_img)
  fft_kernel=np.fft.fft2(new_kernel)
  convolved_img=fft_img * (fft_kernel ** iter)
  convolved_img=np.round(np.fft.ifft2(convolved_img).real)
  return convolved_img

def rescale(img):
  img=img-np.min(img)
  img=img/np.max(img) * 255
  return img


zip_2d = zipfile.ZipFile('/content/drive/MyDrive/cytation Gustavo/2D stitched.zip')
imgs_2d=[img for img in zip_2d.namelist() if img.endswith('.tif')]
column_names = ['picture_id', 'live_cells', 'dead_cells']
df = pd.DataFrame(columns=column_names)

for k in range(int(len(imgs_2d)/2)):
  name= f"{imgs_2d[k*2].split('/')[-2]}_{imgs_2d[k*2].split('/')[-1]}"
  buf_live = zip_2d.read(imgs_2d[k*2])
  img_live = cv2.imdecode(np.frombuffer(buf_live, np.uint8), cv2.IMREAD_GRAYSCALE)
  rescaled=rescale(img_live)
  x_kernel = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]])

  y_kernel = np.array([[1, 0, -1],
                       [1, 0, -1],
                       [1, 0, -1]])

  x_live = convolve(rescaled, x_kernel, 1)[5:-5, 5:-5]
  y_live = convolve(rescaled, y_kernel, 1)[5:-5, 5:-5]
  edge_live = np.sqrt(x_live ** 2 + y_live**2)
  edge_live = rescale(edge_live).astype('uint8')
  thresh_edge = cv2.adaptiveThreshold(edge_live, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                                cv2.THRESH_BINARY, 25, -2)
  thresh_edge=cv2.dilate(thresh_edge, (7,7), iterations=1)

  live_contours, live_hierarchy = cv2.findContours(thresh_edge.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)


  buf_dead = zip_2d.read(imgs_2d[k*2 + 1])
  img_dead = cv2.imdecode(np.frombuffer(buf_dead, np.uint8), cv2.IMREAD_GRAYSCALE)
  img_dead= rescale(img_dead)
  x_dead=convolve(img_dead, x_kernel, 1)[5:-5, 5:-5]
  y_dead=convolve(img_dead, y_kernel, 1)[5:-5, 5:-5]
  edge_dead = np.sqrt(x_dead**2 + y_dead**2)
  edge_dead = rescale(edge_dead)

  ret_dead, thresh_dead = cv2.threshold(edge_dead, 20, 255, cv2.THRESH_BINARY)
  dead_contours, dead_hierarchy = cv2.findContours(thresh_dead.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  dead_cells_list=[]
  for contour in dead_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour)/(w*h) > 0.4:
      dead_cells_list.append(contour)

  live_cells_list=[]
  for contour in live_contours:
    if cv2.contourArea(contour) > 20:
      live_cells_list.append(contour)
      
  column_names = ['picture_id', 'live_cells', 'dead_cells']
  dead_cells=len(dead_cells_list)
  live_cells = len(live_cells_list)
  row_data={'picture_id':name, 'live_cells':live_cells, 'dead_cells':dead_cells}
  row = pd.DataFrame([row_data], columns=column_names)
  df=pd.concat([df, row])

df.to_csv('/content/counts.csv')
