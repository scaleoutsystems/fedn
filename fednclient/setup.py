from setuptools import setup, find_packages

setup(
    name='fednclient',
    version='0.1.6',
    description="""Scaleout Federated Learning""",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    author='Morgan Ekmefjord',
    author_email='morgan@scaleout.se',
    url='https://www.scaleoutsystems.com',
    include_package_data=True,
    py_modules=['fednclient'],
    python_requires='>=3.6,<4',
    install_requires=[
        "fedncommon",
    ],
    license="Copyright Scaleout Systems AB. See license for details",
    zip_safe=False,
    entry_points={
        'console_scripts': ["fednclient=cli:main"]
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
