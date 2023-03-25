from setuptools import setup, find_packages

setup(
    name='lisst',
    version='0.2.1',    
    description='linear shaped skeletons',
    author='Yan Zhang',
    author_email='yan.zhang@inf.ethz.ch',
    license='Apache License 2.0',
    packages=['lisst', 'lisst.models', 'lisst.utils'],
    install_requires=['numpy',
                      'torch>=1.8.0',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: everyone',
        'License :: OSI Approved :: Apache License 2.0',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.8',
    ],
)