from lib.fnc_matching import get_rim, get_reduced

from PySide2 import (QtCore, QtGui, QtPrintSupport)
from collections import defaultdict
from natsort import natsorted
import numpy as np

RIM_GAP = 5
BREAK_GAP = 3
BREAK_LENGTH = 3

def get_lap_descriptors(lap_descriptors = None):
	
	default_lap_descriptors = [
		["Custom_Id",				"Sample.Id"],
		["Profile_Rim",				"Sample.Rim"],
		["Profile_Bottom",			"Sample.Bottom"],
		["Profile_Radius",			"Sample.Radius"],
		["Profile_Geometry",		"Sample.Profile"],
		["Profile_Radius_Point",	"Sample.Radius_Point"],
		["Profile_Rim_Point",		"Sample.Rim_Point"],
		["Profile_Bottom_Point",	"Sample.Bottom_Point"],
		["Profile_Left_Side",		"Sample.Left_Side"],
		["Detail_Geometry",			"Sample.Drawn.Detail.Geometry"],
		["Detail_Closed",			"Sample.Drawn.Detail.Closed"],
		["Detail_Filled",			"Sample.Drawn.Detail.Filled"],
		["Break_Geometry",			"Sample.Drawn.Break.Geometry"],
		["Inflection_Geometry",		"Sample.Drawn.Inflection.Geometry"],
		["Inflection_Dashed",		"Sample.Drawn.Inflection.Dashed"],
	]
	
	def fragments_to_cls_descr(fragments):
		
		fragments = fragments.split(".")
		parent_cls, rel = None, None
		cls, descr = None, None
		if len(fragments) == 2:
			cls, descr = fragments
			parent_cls, rel = None, None
		elif len(fragments) == 4:
			parent_cls, rel, cls, descr = fragments
		return parent_cls, rel, cls, descr
	
	if lap_descriptors is None:
		lap_descriptors = default_lap_descriptors
	lap_descriptors_dict = {}
	for name, fragments in default_lap_descriptors:
		parent_cls, rel, cls, descr = fragments_to_cls_descr(fragments)
		if cls and descr:
			lap_descriptors_dict[name] = [cls, descr]
	return lap_descriptors_dict

def load_drawing_data(store, lap_descriptors, obj_id):
	
	descriptors = dict(
		profile = {}, # {Profile_Geometry, Profile_Rim, Profile_Bottom, Profile_Radius, Profile_Radius_Point, Profile_Rim_Point, Profile_Bottom_Point, Arc_Geometry: []}
		details = defaultdict(dict),  # {target_id: {Detail_Geometry, Detail_Closed, Detail_Filled}, ...}
		inflections = defaultdict(dict),  # {target_id: {Inflection_Geometry, Inflection_Dashed}, ...}
		breaks = defaultdict(dict),  # {target_id: {Break_Geometry}, ...}
	)
	
	prefixes = {
		"Detail_": "details",
		"Inflection_": "inflections",
		"Break_": "breaks",
	}
	
	obj_profile = store.objects[obj_id]
	for name in lap_descriptors:
		if name.startswith("Profile_"):
			descr = lap_descriptors[name][1]
			if descr in obj_profile.descriptors:
				descriptors["profile"][name] = obj_profile.descriptors[descr].label
	
	if lap_descriptors["Custom_Id"][1] not in obj_profile.descriptors:
		return None, None
	
	sample_id = obj_profile.descriptors[lap_descriptors["Custom_Id"][1]].label.value
	
	for rel in obj_profile.relations:
		for target_id in obj_profile.relations[rel]:
			obj2 = store.objects[target_id]
			if lap_descriptors["Custom_Id"][0] in obj2.classes:
				continue
			for name in lap_descriptors:
				cls, descr = lap_descriptors[name]
				if cls not in obj2.classes:
					continue
				if descr not in obj2.descriptors:
					continue
				label = obj2.descriptors[descr].label
				for prefix in prefixes:
					if name.startswith(prefix):
						descriptors[prefixes[prefix]][target_id][name] = label
						break
	return sample_id, descriptors

def get_outer_profile(profile, left_side = False, rim = None, bottom = None):
	
	if rim is None:
		rim = get_rim(profile)
	i1 = np.argmin(((profile - rim)**2).sum(axis = 1))
	if bottom is None:
		i2 = np.where(profile[:,1] == profile[:,1].max())[0]
		i2 = i2[np.argmax(profile[i2,0])]
	else:
		i2 = np.argmin(((profile - bottom)**2).sum(axis = 1))
	if i1 > i2:
		i1, i2 = i2, i1
	profile1 = profile[i1:i2+1]
	profile2 = np.vstack((profile[i2:], profile[:i1+1]))
	
	d = np.abs(profile1[:,1][None,:] - profile2[:,1][:,None])
	r = ((profile1[:,0] > profile2[np.argmin(d, axis = 0),0]).sum() + (profile2[:,0] < profile1[np.argmin(d, axis = 1),0]).sum()) / (profile1.shape[0] + profile2.shape[0])
	if r > 0.5:
		profile = profile2 if left_side else profile1
	else:
		profile = profile1 if left_side else profile2
	if profile[0,1] > profile[-1,1]:
		profile = profile[::-1]
	return profile

def render_drawing(descriptors, painter, linewidth = 1, left_side = False, scale = 1, color = QtCore.Qt.black):
	
	def draw_polygon(coords, closed = True):
		
		if coords is None:
			return
		if closed:
			painter.drawPolygon([QtCore.QPointF(x, y) for x, y in coords])
		else:
			painter.drawPolyline([QtCore.QPointF(x, y) for x, y in coords])
	
	def get_type_value(descriptor, name, type, default):
		
		try:
			return type(descriptor[name].value)
		except:
			return default
	
	def get_coords(descriptor, name, scale, reduce = False):
		
		try:
			coords = np.array(descriptor[name].coords[0])
		except:
			coords = None
		if coords is not None:
			coords *= scale
		if reduce and (coords is not None):
			coords = get_reduced(coords, min(0.1, 0.1 / (scale * 2)))
		return coords
	
	def update_extent(extent, coords):
		
		if coords is None:
			return
		xmin, ymin = coords.min(axis = 0)
		xmax, ymax = coords.max(axis = 0)
		extent[0] = min(xmin, extent[0])
		extent[1] = min(ymin, extent[1])
		extent[2] = max(xmax, extent[2])
		extent[3] = max(ymax, extent[3])
	
	rim_gap = RIM_GAP * scale
	break_gap = BREAK_GAP * scale
	break_length = BREAK_LENGTH * scale
	
	profile_coords = get_coords(descriptors["profile"], "Profile_Geometry", scale, reduce = True)
	rim = get_type_value(descriptors["profile"], "Profile_Rim", int, 0)
	bottom = get_type_value(descriptors["profile"], "Profile_Bottom", int, 0)
	radius = get_type_value(descriptors["profile"], "Profile_Radius", float, 0) * scale
	
	radius_point = get_coords(descriptors["profile"], "Profile_Radius_Point", scale)
	rim_point = get_coords(descriptors["profile"], "Profile_Rim_Point", scale)
	bottom_point = get_coords(descriptors["profile"], "Profile_Bottom_Point", scale)
	
	outer_profile = None
	axis = []
	if radius > 0:
		# construct outer profile
		outer_profile = get_outer_profile(profile_coords, left_side = left_side, rim = rim_point, bottom = bottom_point)
		bottom_xy = outer_profile[-1].copy()
		if left_side:
			outer_profile = outer_profile * [-1,1] + [2*radius, 0]
		else:
			outer_profile = outer_profile * [-1,1] - [2*radius, 0]
		if rim:
			outer_profile = np.vstack(([[rim_gap if left_side else -rim_gap, outer_profile[:,1].min()]], outer_profile))
		if bottom:
			outer_profile = np.vstack((outer_profile, [bottom_xy]))
		
		# construct axis
		top_y = 0
		if rim_point is not None:
			top_y = rim_point[0][1]
		if bottom_point is None:
			bottom_y = profile_coords[:,1].max()
		else:
			bottom_y = bottom_point[0][1]
		if left_side:
			axis_main = np.array([[radius, top_y], [radius, bottom_y]])
		else:
			axis_main = np.array([[-radius, top_y], [-radius, bottom_y]])
		side_mul = 1 if left_side else -1
		if rim:
			whiskers_upper = None
			break_upper = None
		else:
			whiskers_upper = np.array([[side_mul * (radius + break_length / 2), top_y], [side_mul * (radius - break_length / 2), top_y]])
			break_upper = np.array([[side_mul * radius, top_y - break_gap], [side_mul * radius, top_y - break_gap - break_length]])
		if bottom:
			whiskers_lower = None
			break_lower = None
		else:
			whiskers_lower = np.array([[side_mul * (radius + break_length / 2), bottom_y], [side_mul * (radius - break_length / 2), bottom_y]])
			break_lower = np.array([[side_mul * radius, bottom_y + break_gap], [side_mul * radius, bottom_y + break_gap + break_length]])
		axis = [axis_main, whiskers_upper, break_upper, whiskers_lower, break_lower]

	
	details = []
	for obj_id in descriptors["details"]:
		coords = get_coords(descriptors["details"][obj_id], "Detail_Geometry", scale, reduce = True)
		if coords is None:
			continue
		closed = get_type_value(descriptors["details"][obj_id], "Detail_Closed", int, 0)
		filled = get_type_value(descriptors["details"][obj_id], "Detail_Filled", int, 0)
		details.append([coords, closed, filled])
	
	inflections = []
	for obj_id in descriptors["inflections"]:
		coords = get_coords(descriptors["inflections"][obj_id], "Inflection_Geometry", scale)
		if coords is None:
			continue
		dashed = get_type_value(descriptors["inflections"][obj_id], "Inflection_Dashed", int, 0)
		inflections.append([coords, dashed])
	
	breaks = []
	for obj_id in descriptors["breaks"]:
		coords = get_coords(descriptors["breaks"][obj_id], "Break_Geometry", scale)
		if coords is None:
			continue
		breaks.append(coords)
	
	extent = [np.inf, np.inf, -np.inf, -np.inf]
	update_extent(extent, profile_coords)
	update_extent(extent, outer_profile)
	for coords, _, _ in details:
		update_extent(extent, coords)
	for coords, _ in inflections:
		update_extent(extent, coords)
	for coords in breaks:
		update_extent(extent, coords)
	
	profile_coords -= extent[:2]
	if outer_profile is not None:
		outer_profile -= extent[:2]
		for i in range(len(axis)):
			if axis[i] is None:
				continue
			axis[i] -= extent[:2]
	if radius_point is not None:
		radius_point -= extent[:2]
	if rim_point is not None:
		rim_point -= extent[:2]
	if bottom_point is not None:
		bottom_point -= extent[:2]
	for i in range(len(details)):
		details[i][0] -= extent[:2]
	for i in range(len(inflections)):
		inflections[i][0] -= extent[:2]
	for i in range(len(breaks)):
		breaks[i] -= extent[:2]
	
	pen = QtGui.QPen()
	pen.setColor(color)
	pen.setWidth(linewidth)
	painter.setPen(pen)
	painter.setBrush(QtGui.QBrush(color))
	draw_polygon(profile_coords)
	painter.setBrush(QtGui.QBrush())
	if outer_profile is not None:
		draw_polygon(outer_profile, closed = False)
	for coords in axis:
		draw_polygon(coords, closed = False)
	
	for coords, closed, filled in details:
		if filled:
			painter.setBrush(QtGui.QBrush(color))
		else:
			painter.setBrush(QtGui.QBrush())
		draw_polygon(coords, closed = closed)
	
	painter.setBrush(QtGui.QBrush())
	for coords, dashed in inflections:
		if dashed:
			pen.setStyle(QtCore.Qt.DashLine)
		else:
			pen.setStyle(QtCore.Qt.SolidLine)
		painter.setPen(pen)
		draw_polygon(coords, closed = False)
	pen.setStyle(QtCore.Qt.SolidLine)
	painter.setPen(pen)
	
	painter.setBrush(QtGui.QBrush())
	for coords in breaks:
		draw_polygon(coords, closed = False)
	
def save_catalog(path, sample_data, clusters, scale = 1/3, dpi = 600, line_width = 0.5):
	# sample_data = {sample_id: [obj_id, descriptors], ...}
	#	descriptors = {
	#		profile = {Profile_Geometry, Profile_Rim, Profile_Bottom, Profile_Radius, Profile_Radius_Point, Profile_Rim_Point, Profile_Bottom_Point, Arc_Geometry: []}
	#		details = {target_id: {Detail_Geometry, Detail_Closed, Detail_Filled}, ...}
	#		inflections = {target_id: {Inflection_Geometry, Inflection_Dashed}, ...}
	#		breaks = {target_id: {Break_Geometry}, ...}
	#	}
	# clusters = {label: [sample_id, ...], ...}
	
	def get_picture(sample_id, descriptors, scale, line_width):
		
		picture = QtGui.QPicture()
		painter = QtGui.QPainter(picture)
		render_drawing(descriptors, painter, line_width, scale = scale)
		painter.end()
		return picture
	
	def init_cluster(td, painter, cluster_label, y):
		
		td.setHtml("Cluster: %s" % (cluster_label))
		h = td.size().height() + 24
		painter.translate(0, y)
		td.drawContents(painter)
		painter.translate(0, -y)
		
		return h
	
	labels = natsorted(clusters.keys())
	sample_ids = set([])
	for label in labels:
		for sample_id in clusters[label]:
			sample_ids.add(sample_id)
	cmax = len(sample_ids) + len(clusters)
	cnt = 1
	drawings = {}
	for sample_id in sample_ids:
		print("\rrendering %d/%d            " % (cnt, cmax), end = "")
		cnt += 1
		drawings[sample_id] = get_picture(sample_id, sample_data[sample_id][1], scale, line_width)
	
	printer = QtPrintSupport.QPrinter()
	printer.setWinPageSize(QtGui.QPageSize.A4)
	printer.setResolution(dpi)
	printer.setOrientation(QtPrintSupport.QPrinter.Portrait)
	printer.setOutputFormat(QtPrintSupport.QPrinter.PdfFormat)
	printer.setOutputFileName(path)
	w_max = printer.width()
	h_max = printer.height()
	
	painter = QtGui.QPainter(printer)
	
	td = QtGui.QTextDocument()
	font = td.defaultFont()
	font.setPointSize(24)
	td.setDefaultFont(font)
	
	x = 0
	y = 0
	h_max_row = 0
	
	for label in labels:
		print("\rrendering %d/%d            " % (cnt, cmax), end = "")
		cnt += 1
		
		x = 0
		y += h_max_row
		if h_max - y > 200:
			y += init_cluster(td, painter, label, y)
		else:
			printer.newPage()
			y = init_cluster(td, painter, label, 0)
		h_max_row = 0
		
		for sample_id in clusters[label]:
			
			rect = drawings[sample_id].boundingRect()
			mul = printer.resolution() / drawings[sample_id].logicalDpiX()
			w = (rect.width() * mul) * 1.2
			h = (rect.height() * mul) * 1.2
			h_max_row = max(h_max_row, h)
			
			if x + w > w_max:
				x = 0
				y += h_max_row
				h_max_row = h
			
			if y + h_max_row > h_max:
				printer.newPage()
				x = 0
				y = init_cluster(td, painter, label, 0)
			
			painter.drawPicture(x, y, drawings[sample_id])
			x += w
	
	painter.end()


