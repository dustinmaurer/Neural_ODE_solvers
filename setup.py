from setuptools import setup, find_packages

setup(
    name='neural-ode',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='GNU General Public License v3.0',
    description='A package to solve for ODE parameters with with back-propogation of errors',
    long_description=open('README.md').read(),
    install_requires=['numpy','scipy'],
    url='https://github.com/dustinmaurer/Neural_ODE_solvers',
    author='Dustin Maurer',
    author_email='maurer.dustin@gmail.com'
)