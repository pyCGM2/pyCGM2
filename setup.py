from setuptools import setup,find_packages
import os

def gen_data_files(*dirs):
    results = []

    for src_dir in dirs:
        for root,dirs,files in os.walk(src_dir):
            results.append((root, map(lambda f:root + "/" + f, files)))
    return results

setup(name = 'pyCGM2',
    version = '0.0.1',
    author = 'fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python Conventional Gait Model',
    packages=find_packages(),
	data_files = gen_data_files("Apps","Data","Extern","NoViconApps","Session Settings","third party"),
	include_package_data=True,
	install_requires = ['numpy>=1.11.0', 
                        'scipy>=0.17.0',
                        'matplotlib>=1.5.3',
                        'pandas >=0.19.1',
                        'enum34>=1.1.2']
     )
