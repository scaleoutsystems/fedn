from setuptools import setup, find_packages

setup(
    name='fedn',
    version='0.2.5',
    description="""Scaleout Federated Learning""",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Morgan Ekmefjord',
    author_email='morgan@scaleout.se',
    url='https://www.scaleoutsystems.com',
    py_modules=['fedn'],
    python_requires='>=3.6,<4',
    install_requires=[
        "PyYAML>=5.4",
        "requests",
        "urllib3>=1.26.4",
        "minio",
        "python-slugify",
        "grpcio-tools",
        "grpcio~=1.34.0",
        "numpy~=1.19.5",
        "protobuf",
        "pymongo",
        "Flask",
        "Flask-WTF",
        "pyopenssl",
        "ttictoc",
        "psutil",
        "click==8.0.1",
        "jinja2<3.0,>=2.10.1",
        "geoip2",
        "plotly",
        "pandas",
        "bokeh",
        "networkx"
    ],
    license="Copyright Scaleout Systems AB. See license for details",
    zip_safe=False,
    entry_points={
        'console_scripts': ["fedn=cli:main"]
    },
    keywords='Federated learning',
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
