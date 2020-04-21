from setuptools import find_namespace_packages, setup

MODULE_NAME = "pyodi"
PACKAGE_NAME = "pyodi"
VERSION = "0.0"

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    packages=find_namespace_packages(
        include="{}.*".format(MODULE_NAME), exclude=["tests", "logs"]
    ),
    include_package_data=True,
    platforms="any",
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'pyodi = pyodi.cli:app',
        ],
    },
)
