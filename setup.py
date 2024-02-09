from setuptools import find_packages, setup

with open("requirements.txt", encoding="utf-8") as fp:
    install_requires = fp.read().split("\n")

DATASET_FILES = [
    "events/ncaltech101/caltech101_labels.txt",
    "**/checksums.tsv",
]

setup(
    name="events-tfds",
    version="0.1",
    description="tensorflow_datasets implementations for events datasets",
    url="http://github.com/jackd/events-tfds",
    author="Dominic Jack",
    author_email="thedomjack@gmail.com",
    license="MIT",
    packages=find_packages(),
    requirements=install_requires,
    include_package_data=True,
    package_data={"events_tfds": DATASET_FILES},
)
