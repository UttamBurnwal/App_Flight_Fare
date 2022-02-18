# App_Flight_Fare:
--> A Machine Learning Web App that can predict the flight prices. It is created with Flask and deployed on Heroku platform.

Visit: https://flight-fare-generator.herokuapp.com/

## A glimpse of the web app:

 ![GIF](readme_resources/ipl-first-innings-score-web-app.gif)
 

## Directory Tree
--> Here are some details of the subdirectories and files that the repository contains. 
```
├── static 
│   ├── flights-favicon.ico
│   ├── styles.css
├── template
│   ├── home.html
├── Data_Train.xlsx
├── Flight_fare_prediction.py
├── Procfile
├── README.md
├── app.py
├── flight_price_rfr.pkl
├── requirements.txt
```
## Description
* Static Folder contains the favicon as well as the css file which describes the HTML elements of the web app.
* Template folder contains the html file which depicts the format of the web page.
* Data_Train.xlsx is the dataset file which is the excel format.
* Flight_fare_prediction.py contains all the notebook code, the execution results as well as the codes that has helped in generating the model for the web app.
* Procfile includes the code ``` web: gunicorn app:app --preload ``` which depicts that gunicorn commands are run by the application's containers on the platform. To create a procfile run the following command on command prompt. ``` echo web: gunicorn app:app --preload > Procfile ```
* README.md includes the stucture that provides a detailed description of my GitHub project.
* app.py includes the all the routes and functions to perform the actions of web app. This file is the root of our Flask application which we will run in the command line prompt.
* flight_price_rfr.pkl is the random forest regression model that is the core of my web application.
* Requirements.txt file includes all the libraries that has been used to create the web app. To create a requirments.txt file, run the following command in command prompt.``` pip freeze > requirements.txt ```

## Libraries Used
--> This section contains the list of the libraries that have been used to create the web app. 
```
certifi==2020.6.20
click==8.0.3
colorama==0.4.4
Flask==2.0.3
Flask-Cors==3.0.10
itsdangerous==2.0.1
Jinja2==3.0.3
joblib==1.1.0
MarkupSafe==2.0.1
numpy==1.22.2
pandas==1.4.1
python-dateutil==2.8.2
pytz==2021.3
scikit-learn==1.0.2
scipy==1.8.0
six==1.16.0
sklearn==0.0
threadpoolctl==3.1.0
Werkzeug==2.0.3
wincertstore==0.2
gunicorn
```



