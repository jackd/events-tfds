from setuptools import setup
from setuptools import find_packages

with open('requirements.txt') as fp:
    install_requires = fp.read().split('\n')

DATASET_FILES = ['url_checksums/*', 'events/caltech101_labels.txt']

setup(name='events-tfds',
      version='0.1',
      description='tensorflow_datasets implementations for events datasets',
      url='http://github.com/jackd/events-tfds',
      author='Dominic Jack',
      author_email='thedomjack@gmail.com',
      license='MIT',
      packages=find_packages(),
      requirements=install_requires,
      include_package_data=True,
      package_data={'events_tfds': DATASET_FILES}
      # zip_safe=False
     )
