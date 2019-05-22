from PySide2 import (QtWidgets, QtCore, QtGui, QtSvg, QtPrintSupport)
from collections import defaultdict
from natsort import natsorted
from deposit.store.Conversions import as_path

def save_clusters_as_pdf(tgt_path, samples):
	
	clusters = defaultdict(list) # {label: [sample, ...], ...}
	for sample in samples:
		if sample.cluster is None:
			continue
		clusters[sample.cluster].append(sample)
	if not clusters:
		return
	for cluster in clusters:
		clusters[cluster] = natsorted(clusters[cluster], key = lambda sample: sample.leaf)
	
	renderer = QtSvg.QSvgRenderer()
	w_src_max = 0
	h_src_max = 0
	cmax = 0
	for cluster in clusters:
		for sample in clusters[cluster]:
			cmax += 1
			src_path = as_path(sample.resource.value, check_if_exists = False)
			renderer.load(src_path)
			rnd_size = renderer.defaultSize()
			w_src_max = max(w_src_max, rnd_size.width())
			h_src_max = max(h_src_max, rnd_size.height())
	
	printer = QtPrintSupport.QPrinter()
	printer.setWinPageSize(QtGui.QPageSize.A4)
	printer.setResolution(300)
	printer.setOrientation(QtPrintSupport.QPrinter.Portrait)
	printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
	printer.setOutputFileName(tgt_path)
	
	w_max = printer.width()
	h_max = printer.height()
	
	scale = min(1, w_max / w_src_max, h_max / h_src_max)
	
	painter = QtGui.QPainter(printer)
	
	td = QtGui.QTextDocument()
	font = td.defaultFont()
	font.setPointSize(36)
	td.setDefaultFont(font)
	
	def new_page(printer, td, painter, cluster, page):
		
		printer.newPage()
		td.setHtml("Cluster: %s, Page: %s" % (cluster, page))
		td.drawContents(painter)
		return td.size().height()
	
	cnt = 0
	for cluster in clusters:
		x = 0
		y = 0
		h_max_row = 0
		page = 1
		
		y = new_page(printer, td, painter, cluster, page)
		
		for sample in clusters[cluster]:
			print("\rgen. pdf %d/%d            " % (cnt, cmax), end = "")
			cnt += 1
			
			src_path = as_path(sample.resource.value, check_if_exists = False)
			renderer.load(src_path)
			rnd_size = renderer.defaultSize()
			w = rnd_size.width() * scale
			h = rnd_size.height() * scale
			h_max_row = max(h_max_row, h)
			
			if x + w > w_max:
				x = 0
				y += h_max_row
				h_max_row = h
			
			if y + h_max_row > h_max:
				x = 0
				page += 1
				y = new_page(printer, td, painter, cluster, page)
			
			renderer.render(painter, QtCore.QRectF(x, y, w, h))
			x += w

	painter.end()

