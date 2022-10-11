import multiprocessing as mp
import time

def _worker(worker_fnc, params_mp, collect_mp, max_queue_size, args):
	
	while True:
		try:
			params = params_mp.get()
		except:
			time.sleep(0.01)
			continue
		
		while collect_mp.qsize() > max_queue_size:
			time.sleep(0.01)
			pass
		collect_mp.put(worker_fnc(params, *args))

def process_mp(worker_fnc, params_list, worker_args = [], collect_fnc = None, collect_args = [], progress_fnc = None, progress_args = [], max_cpus = -1, max_queue_size = 100):
	
	def call_progress(progress_fnc, done, todo, progress_args):
		
		if progress_fnc == True:
			print("\r%d/%d             " % (done, todo), end = "")
		elif callable(progress_fnc):
			return progress_fnc(done, todo, *progress_args)
		return True
	
	params_mp = mp.Queue()
	for params in params_list:
		params_mp.put(params)
	todo = len(params_list)
	done = 0
	collect_mp = mp.Queue(todo)
	if max_cpus > 0:
		n_cpus = min(max_cpus, mp.cpu_count() - 1, todo)
	else:
		n_cpus = min(mp.cpu_count() - 1, todo)
	call_progress(progress_fnc, done, todo, progress_args)
	procs = []
	while done < todo:
		if len(procs) < n_cpus:
			procs.append(mp.Process(target = _worker, args = (worker_fnc, params_mp, collect_mp, max_queue_size, worker_args)))
			procs[-1].start()
		if not collect_mp.empty():
			data = collect_mp.get()
			done += 1
			if not call_progress(progress_fnc, done, todo, progress_args):
				break
			if collect_fnc is not None:
				collect_fnc(data, *collect_args)
		else:
			time.sleep(0.5)
	for proc in procs:
		proc.terminate()
		proc = None
