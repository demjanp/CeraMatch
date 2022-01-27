rmdir dist\cm /S /Q

pyinstaller cm.spec --hidden-import="sklearn.utils._cython_blas" --hidden-import="sklearn.neighbors.typedefs" --hidden-import="sklearn.neighbors.quad_tree" --hidden-import="sklearn.tree._utils" --hidden-import="sklearn.neighbors._typedefs" --hidden-import="sklearn.utils._weight_vector" --hidden-import="sklearn.neighbors._quad_tree" --hidden-import="skimage.filters.rank.core_cy_3d"

robocopy /e res dist\cm\res
robocopy /e deposit dist\cm\deposit
robocopy /e graphviz dist\cm\graphviz
robocopy /e pygraphviz dist\cm\pygraphviz
copy cm_icon.ico dist\cm
copy LICENSE dist\cm
copy THIRDPARTY.TXT dist\cm
python make_ifp.py
