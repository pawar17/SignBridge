"""
SignBridge Setup Configuration
"""
from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='signbridge',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Bidirectional Sign Language Translation and Learning Platform',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/SignBridge',
    packages=find_packages(exclude=['tests', 'notebooks', 'scripts']),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Video',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.7.0',
            'flake8>=6.0.0',
            'mypy>=1.4.0',
            'pre-commit>=3.3.0',
        ],
        'docs': [
            'mkdocs>=1.5.0',
            'mkdocs-material>=9.1.0',
            'mkdocstrings[python]>=0.22.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'signbridge-train=scripts.training.train_model:main',
            'signbridge-evaluate=scripts.training.evaluate_model:main',
            'signbridge-server=backend.api.main:run_server',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='sign-language machine-learning computer-vision accessibility nlp deep-learning',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/SignBridge/issues',
        'Source': 'https://github.com/yourusername/SignBridge',
        'Documentation': 'https://signbridge.readthedocs.io/',
    },
)
