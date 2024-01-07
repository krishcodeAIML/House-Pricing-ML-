house_pricing
==============================

Modelbuild to house pricing

Project Organization
------------

    ├── LICENSE
    ├── README.md                	<- The top-level README document for project structure and usage of scripts
    ├── data
    │   ├── analysis_results    	<- All the initial data analysis, descriptive analysis and correlation analysis results are stored here. Check for null values and data types etc. 
    │   └── raw            		   	<- The original raw data location
    │
    ├── models             		   	<- Trained and serialized models saved here
    │
    ├── results				<- Final model metrics for different models saved here
    │
    ├── reports 
    │   └── figures        	   	<- Generated graphics and figures to be used in reporting. Contains data distribution  and correlation scatter plots
    │	 └── Modelreport_summary.docx    	<- Word document containing report summary and analysis for the given prediction problem.
    ├── requirements.txt   	    	<- The requirements file for reproducing the analysis environment, 
    │
    ├── setup.py                       	<- contains project details
    │
    ├── src                		   	<- Source code for use in this project.
    │   ├── __init__.py                	<- Makes src a Python module
    │   │
    │	 ├── main_orch.py   		<- The main script starts here. Orchestrates data and model pipeline	
    │   │
    │   ├── data_prep          		<- Scripts to download and prepare data for model training
    │   │   └── data_main.py	<- Script for orchestrating data preparation pipeline
    │   │   └── dataanalysis_vis.py	<- Script for creating descriptive analysis, correlation analysis and  distribution & correlation scatter plots
    │   │   └── data_settings.py   <- All data configurations related to data preparation saved in this file
    │   │   └── data_split.py   	<- Script that splits the data to train and test
    │   │   └── data_transform.py  	<- All the data preparation code (Ex: normalizaion, outlier removal, categorical variable encoding) to model required format
    │   │
    │   ├── models
    │   │   ├── model_settings.py 	<- All model configurations related to model training saved in this file
    │   │   └── train_model.py	<- All model training code (model training, save, prediction and model metrics logic) saved here




--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
