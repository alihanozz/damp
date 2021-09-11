import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='damp',  
     version='0.1',
     scripts=['damp'] ,
     author="Data Analyser/Manipulator/Processor/",
     author_email="alihanozmm@gmail.com",
     description="A pre-machine-learning model package",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/alihanozz/data_analyzer",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )