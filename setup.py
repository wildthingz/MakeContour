from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md')) as f:
	long_description = f.read()

setup(name = 'MakeContour',
	version = '0.1.0',
	description = 'Python package for visualising macrosegregation in a micorstructure image montage.',
	long_description = long_description,
	url = 'https://github.com/wildthingz/MakeContour',
	author = 'Hatef Khadivinassab',
	author_email = 'hatef.hadivinassab@gmail.com',
	packages = ['MakeContour']
	classifiers=[
		"Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Operating System :: Linux :: Linux Debian"
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
		'Programming Language :: Python :: 2.7',
		'Framework :: Spyder',
		'Intended Audience :: End Users/Desktop',
		'Natural Language :: English',
		],
	license = 'Creative Commons Attribution-Noncommercial-Share Alike license',
	keywords = ['macrosegregation', 'mesh', 'visaliziation', 'contour']

	)
