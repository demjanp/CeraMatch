# -*- coding: utf-8 -*-
#!/usr/bin/env python
#

from setuptools import setup, find_packages
import pathlib

try:
	from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
	class bdist_wheel(_bdist_wheel):
		def finalize_options(self):
			_bdist_wheel.finalize_options(self)
			self.root_is_pure = False
except ImportError:
	bdist_wheel = None

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
	name="ceramatch",
	version="1.1.1",
	description="Visual shape-matching and classification of ceramics.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/demjanp/CeraMatch",
	author="Peter Demján",
	author_email="peter.demjan@gmail.com",
	classifiers=[
		"Development Status :: 5 - Production/Stable",
		"Intended Audience :: Science/Research",
		"Topic :: Scientific/Engineering",
		"License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
		"Programming Language :: Python :: 3.10",
		"Operating System :: Microsoft :: Windows :: Windows 10",
	],
	keywords="ceramics, classification, cluster analysis, morphometrics",
	package_dir={"": "src"},
	packages=find_packages(where="src"),
	include_package_data=True,
	package_data={"":[
		"LICENSE",
		"*.TXT",
		"*.pyd",
		"*.png",
		"*.svg",
	]},
	python_requires=">=3.10, <4",
	install_requires=[
		'lap_data',
		'scipy',
		'scikit-image',
		'scikit-learn',
		'opencv-python',
		'numpy',
	],
	cmdclass={'bdist_wheel': bdist_wheel},
)
