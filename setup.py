try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(
    name='see-segment',
    author='Dirk Colbry',
    author_email='colbrydi@msu.edu',
    url='http://see-segment.github.io/see-segment',
    version='0.1.0',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires = [
	'numpy',
	'matplotlib',
  	'scikit-image',
  	'argparse',
  	'deap',
  	'opencv-python',
  	'scoop',
    ],    
    packages=[
        'see',
        'see.tests',
    ],
)
