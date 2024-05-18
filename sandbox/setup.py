from setuptools import setup, find_packages

setup(
    name="sandbox",
    version='1.0',
    author='soulde',
    install_requires=['numpy', 'opencv-python'],
    # 你要安装的包，通过 setuptools.find_packages 找到当前目录下有哪些包
    packages=find_packages()
)
