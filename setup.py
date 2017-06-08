from setuptools import setup, find_packages

setup(name='caimcaim',
      version='0.1',
      description='CAIM',
      long_description='CAIM Discretization Algorithm',
      classifiers=[
          'Development Status :: Alpha',
          'License :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Supervised discretization',
      ],
      url='https://github.com/airysen/noisePGA',
      author='Arseniy Kustov',
      author_email='me@airysen.co',
      license='MIT',
      packages=find_packages(),
      install_requires=['numpy', 'sklearn'],
      include_package_data=True,

      zip_safe=False)
