from deposit.store.Store import Store
from lib.fnc_matching import *

store = Store()
store.load("d:\\documents_synced\\CeraMatch\\DEP_kap_typology\\kap_typology.json")

cmax = len(store.classes["Sample"].objects)
cnt = 1
for id in store.classes["Sample"].objects:
	print("\r%d/%d               " % (cnt, cmax), end = "")
	cnt += 1
	obj = store.objects[id]
	profile = np.array(obj.descriptors["Profile"].label.coords[0])
	axis, thickness = profile_axis(profile, rasterize_factor = 10)
	axis_length = np.sqrt((np.diff(axis, axis = 0) ** 2).sum(axis = 1)).sum()
	obj.add_descriptor("Axis_Length", axis_length)
	obj.add_descriptor("Norm_Length", axis_length / thickness)
	
store.save()
