from deposit import Store

from lib.fnc_matching import *
import json
import numpy as np

fprofiles = "data/profiles.json"

store = Store()
store.load("c:\\Documents\\AE_KAP_morpho\\db_typology\\kap_typology.json")

id_lookup = {} # {Sample.Id: obj_id, ...}
for id in store.classes["Sample"].objects:
	id_lookup[store.objects[id].descriptors["Id"].label.value] = id

with open(fprofiles, "r") as f:
	profiles = json.load(f)

for sample_id in profiles:
	store.objects[id_lookup[sample_id]].add_descriptor("Perimeter", profile_length(np.array(profiles[sample_id]["profile"])))

store.save()
