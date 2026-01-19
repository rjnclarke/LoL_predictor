# Code Structure

The app is run in streamlit by app.py
The visualization run from the data folder by, and use a simple entities schema in core
The ml folder defines details and code relavent to the Transformer prediction model
Crawler contains all the logic for the app
  1. db_handler - database interface
  2. riot_api_client - handles api calls
  3. match_base - interaction between api and database
  4. data_collector - collects match ids for ml model
  5. feature_builder - builds the dataset for the ml model


**The app will not run without first collection of data for the model (requires 1+ days and API key) and creation of the database for the visualizations.**
