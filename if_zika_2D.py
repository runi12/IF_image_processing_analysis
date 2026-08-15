zip_2d = zipfile.ZipFile('/content/drive/MyDrive/cytation Gustavo/2D stitched.zip')
imgs_2d=[img for img in zip_2d.namelist() if img.endswith('.tif')]
column_names = ['picture_id', 'live_cells', 'dead_cells']
#os.mkdir('/content/2D_stitched')
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
  #img_dead=img_dead.astype('uint8')
  #dead_values, dead_count = np.unique(img_dead, return_counts=True)
  #plt.plot(dead_values, dead_count)
  #plt.show()

  ret_dead, thresh_dead = cv2.threshold(edge_dead, 20, 255, cv2.THRESH_BINARY)
  #thresh_dead=cv2.adaptiveThreshold(img_dead.astype('uint8'), 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 25, -5)
  dead_contours, dead_hierarchy = cv2.findContours(thresh_dead.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

  blue = np.zeros(img_dead.shape).astype('uint8')[5:-5, 5:-5]
  merged= cv2.merge([blue, img_live[5:-5, 5:-5], img_dead.astype('uint8')[5:-5, 5:-5]])
  dead_cells_list=[]
  for contour in dead_contours:
    x, y, w, h = cv2.boundingRect(contour)
    if cv2.contourArea(contour)/(w*h) > 0.4:
      cv2.drawContours(merged, contour, -1, (125,125,255), 1)
      dead_cells_list.append(contour)
  #[cv2.drawContours(merged, contour, -1, (255,255, 255), 1) for contour in dead_contours if cv2.contourArea(contour)]
  live_cells_list=[]
  for contour in live_contours:
    if cv2.contourArea(contour) > 20:
      live_cells_list.append(contour)
      cv2.drawContours(merged, contour, -1, (255, 125, 125), 1)
  #cv2.drawContours(merged, live_cells_list, -1, (255, 125, 125), 1)
  column_names = ['picture_id', 'live_cells', 'dead_cells']
  dead_cells=len(dead_cells_list)
  live_cells = len(live_cells_list)
  row_data={'picture_id':name, 'live_cells':live_cells, 'dead_cells':dead_cells}
  row = pd.DataFrame([row_data], columns=column_names)
  df=pd.concat([df, row])
  merged = cv2.resize(merged, (0, 0), fx = 0.5, fy = 0.5)
  cv2.imwrite(f'/content/2D_stitched/{name}', merged)
