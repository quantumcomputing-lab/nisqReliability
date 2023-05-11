# nisqReliability

Step 1: 
Use create_db.py
Update the start and stop dates. 
Create the new data file. 
This will be merged with existing database in next step.

Step 2:
Use update_db.py
This will join the database

Step 3:
Use washington_historical_database_monthly.py to create the monthly files

Step 4:
Run config file to set program configuration

Step 5:
Load the statistical and quantum utility functions. 
For the latter, you might need to connect to IBM using load_account().
This will require your token, password etc.

Step 6:
Consult the List of figures and tables and generate what you need

