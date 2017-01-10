from setuptools import setup
setup(name = 'pyCGM2',
    version = '0.0.1',
    author = 'fabien Leboeuf',
    author_email = 'fabien.leboeuf@gmail.com',
    keywords = 'python Conventional Gait Model',
    packages = ['pyCGM2'],
    install_requires = ['numpy>=1.11.0', 
                        'scipy>=0.17.0',
                        'matplotlib>=1.5.3',
                        'pandas >=0.19.1',
                        'enum34>=1.1.2']
     )