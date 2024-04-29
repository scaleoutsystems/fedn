from setuptools import find_packages, setup

setup(
    name='fedn',
    version='0.9.1',
    description="""Scaleout Federated Learning""",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Scaleout Systems AB',
    author_email='contact@scaleoutsystems.com',
    url='https://www.scaleoutsystems.com',
    py_modules=['fedn'],
    python_requires='>=3.8,<3.12',
    install_requires=[
        "requests",
        "urllib3>=1.26.4",
        "minio",
        "grpcio~=1.60.0",
        "grpcio-tools~=1.60.0",
        "numpy>=1.21.6",
        "protobuf~=4.25.2",
        "pymongo",
        "Flask==3.0.3",
        "pyjwt",
        "pyopenssl",
        "psutil",
        "click==8.1.7",
        "grpcio-health-checking~=1.60.0",
        "pyyaml",
        "plotly",
        "virtualenv",
        "opentelemetry-api",
        "opentelemetry-sdk",
        "opentelemetry-exporter-jaeger",
        "opentelemetry-exporter-otlp",
    ],
    extras_require={
        'flower': ["flwr==1.8.0"]
    },
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
        'Programming Language :: Python :: 3.11',
    ],
)
