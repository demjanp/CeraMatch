from deposit import Store

import json

store = Store()
store.load("c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json")

for id in store.classes["Sample"].objects:
	obj = store.objects[id]
	perimeter = float(obj.descriptors["Perimeter"].label.value)
	thickness = float(obj.descriptors["Thickness"].label.value)
	obj.add_descriptor("Norm_Length", round((perimeter / 2) / thickness, 4))

store.save()
