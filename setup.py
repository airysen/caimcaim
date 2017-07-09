from setuptools import setup, find_packages

setup(name='caimcaim',
      version='0.3',
      description='CAIM',
      long_description='CAIM Discretization Algorithm',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
      ],
      url='https://github.com/airysen/caimcaim',
      author='Arseniy Kustov',
      author_email='me@airysen.co',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'sklearn'],
      include_package_data=True,

      zip_safe=False)
