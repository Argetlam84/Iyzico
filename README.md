###############################################################
# Business Problem
###############################################################

# Iyzico is a financial technology company that facilitates the online shopping experience for both buyers and sellers.
# It provides payment infrastructure for e-commerce companies, marketplaces, and individual users.
# It is expected to make total transaction volume predictions by merchant_id and on a daily basis for the last 3 months of 2020.


###############################################################
# Dataset Story
###############################################################
# The data includes the records of 7 member merchants from 2018 to 2020.

# Transaction: Number of transactions
# MerchantID: IDs of member merchants
# Paid Price: Payment amount

###############################################################
# TASKS
###############################################################

# Task 1: Exploration of the Dataset
            # 1. Read the ts-iyzico_data.csv file. Convert the transaction_date variable to date type.
            # 2. What are the start and end dates of the dataset?
            # 3. How many total transactions are there for each member merchant?
            # 4. What is the total payment amount for each member merchant?
            # 5. Observe the transaction count graphs for each member merchant for each year.

# Task 2: Apply Feature Engineering techniques. Derive new features.
            # Date Features
            # Lag/Shifted Features
            # Rolling Mean Features
            # Exponentially Weighted Mean Features

# Task 3: Preparation for Modeling
            # 1. Perform One-hot encoding.
            # 2. Define Custom Cost Functions.
            # 3. Split the dataset into train and validation sets.

# Task 4: Create a LightGBM Model and observe the error value with SMAPE.

