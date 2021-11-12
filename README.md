**CONTENTS**
=======================
* INTRODUCTION
* REPOSITORY
* CONFIGURATION

**INTRODUCTION**
================
A Github repository for Group 2's final capstone project. The project looks at financial stock data of six companies and the S&P 500 Index and sees how the prices change over time. A machine learning model is created to determine if it is possible to predict if a stock will outperform the S&P 500 on a given day.

**REPOSITORY**
==============
The repository includes a Code folder with all the relevant code formatted in ipynb, python, or sql files with a single pickle file also included. The pickle file for the machine learning model in the dashboard is named "Capstone_ML2". 

The repository also includes a Project Specifications folder with the data platform, a list of data sources, the executive summary, and project plan all in PDF format.

Other deliverables are included in the github as PDF files. The Power BI version of the dashboard is contained as "index.md" and can be viewed <a href="https://boutdara.github.io/capstone/">here</a> (with permissions).

**CONFIGURATION**
=================
To navigate the Code folder in the repository, the files "Capstone Data Setup", "Capstone Producer", "Capstone Consumer", and "Capstone SQL" were all exported from an Azure Data Brick Notebook into a Jupyter Notebook format. The "Capstone Data Setup" arranges the data into a format to be used for this project and will send the data to the data lake where the Producer will pull the data to send it along the producer stream where the Consumer file will read the messages. The "Capstone Consumer" will then send the messages as a dataframe to a folder within the data lake titled "pipeline". The "Capstone SQL" file is meant to be run after the consumer file has pushed data to the data lake container. These files can be run as Jupyter Notebook files using Python code. 

The file "capstone.sql" was created in Azure Data Studio as a SQL file and is to be run once the data has been sent to the SQL server.

The files "Capstone Machine Learning" and "Capstone ML Load" were created in Jupyter Notebook. The "Capstone Machine Learning" file has the creation of the ML model that is exported as a pickle object to be loaded in the "Capstone ML Load" file. Both files load their data through the SQL server database.

The dashboard "Capstone Dashboard" is a python file that will need the pickle file "Capstone_ML2" to be downloaded in order to run the machine learning model. The pickle file can also be created by running the "Capstone Machine Learning" file in its entirety.


```python

```
