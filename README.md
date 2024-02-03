# Housing Price Prediction Model

This project utilizes a linear regression model to predict housing prices based on various features. The dataset used is contained in `konut.csv`, which includes multiple variables related to housing such as size, location, age, and more.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.11
- pandas
- scikit-learn

You can install the necessary libraries using pip:

```bash
pip install pandas scikit-learn
git clone https://github.com/merttunayilmaz/PricePrediction.git
cd PricePrediction

Install the required dependencies:
pip install -r requirements.txt

Usage
To run the model, execute the main script:
python main.py

This script will perform the following actions:

Load the housing data from Data/konut.csv.
Split the data into independent (X) and dependent (Y) variables, excluding 'Price' and 'Port_no' from X.
Normalize the independent variables using MinMaxScaler.
Train a linear regression model and evaluate it using the Mean Absolute Error (MAE) metric.
Split the data into training and test sets and re-evaluate the model.
Contributing
Please read CONTRIBUTING.md for details on our code of conduct, and the process for submitting pull requests to us.

License
This project is licensed under the Apache 2.0 - see the LICENSE.md file for details.

Acknowledgments
Hat tip to anyone whose code was used
Inspiration
etc.

