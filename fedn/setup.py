from setuptools import find_packages, setup

setup(
    name='fedn',
    version='0.8.0',
    description="""Scaleout Federated Learning""",
    author='Scaleout Systems AB',
    author_email='contact@scaleoutsystems.com',
    url='https://www.scaleoutsystems.com',
    py_modules=['fedn'],
    python_requires='>=3.8,<3.11',
    install_requires=[
        "requests",
        "urllib3>=1.26.4",
        "minio",
        "grpcio~=1.57.0",
        "grpcio-tools~=1.57.0",
        "numpy>=1.21.6",
        "protobuf",
        "pymongo",
        "Flask",
        "pyjwt",
        "pyopenssl",
        "psutil",
        "click==8.0.1",
        "grpcio-health-checking~=1.57.0",
        "flasgger==0.9.5",
        "plotly",
    ],
    license='Apache 2.0',
    zip_safe=False,
    entry_points={
        'console_scripts': ["fedn=cli:main"]
    },
    keywords='Federated learning',
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
