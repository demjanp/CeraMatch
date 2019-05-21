import os
from PySide2 import (QtWidgets, QtCore, QtGui, QtSvg)
from PIL import Image
import numpy as np
import sys

dir_images = "data\\db_typology\\0000"

scales = [1, 1/2, 1/4]
thumb_size = 512

app = QtWidgets.QApplication(sys.argv)

paths = []
for fname in os.listdir(dir_images):
	src_path = os.path.join(dir_images, fname)
	paths.append([src_path, fname.split(".")[0]])

max_w, max_h = 0, 0
min_w, min_h = np.inf, np.inf
for src_path, _ in paths:
	renderer = QtSvg.QSvgRenderer(src_path)
	rnd_size = renderer.defaultSize()
	w, h = rnd_size.width(), rnd_size.height()
	max_w, max_h = max(max_w, w), max(max_h, h)
	min_w, min_h = min(min_w, w), min(min_h, h)

max_scale = thumb_size / (max(max_w, max_h) * min(scales))
scales = [scale * max_scale for scale in scales]

cmax = len(paths)
cnt = 1
for src_path, name in paths:
	
	if cnt % 10 == 0:
		print("\r%d/%d      " % (cnt, cmax), end = "")
	cnt += 1
	
	tgt_path = os.path.join("data", "thumbnails", "%s.jpg" % (name))
	
	renderer = QtSvg.QSvgRenderer(src_path)
	rnd_size = renderer.defaultSize()
	w, h = rnd_size.width(), rnd_size.height()
	size_max = max(w, h)
	for scale in scales:
		if size_max * scale <= thumb_size:
			break
	w, h = int(round(w * scale)), int(round(h * scale))
	x, y = (thumb_size - w) / 2, (thumb_size - h) / 2
	pixmap = QtGui.QPixmap(thumb_size, thumb_size)
	pixmap.fill(QtCore.Qt.white)
	painter = QtGui.QPainter(pixmap)
	renderer.render(painter, QtCore.QRectF(x, y, w, h))
	pixmap.save(tgt_path)
	painter.end()

