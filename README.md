# car_dheko_machine_learning
## 1.Introduction
The Car Dekho Project is a comprehensive data-driven machine learning solution aimed at predicting 
used car prices based on various features such as mileage, engine capacity, fuel type, and more. 
Leveraging a dataset of used cars, the project involved detailed preprocessing, feature engineering, and 
building an efficient regression model to accurately estimate car prices. The goal was to create a robust 
prediction model that would assist users in making informed decisions about buying or selling used cars, 
providing a transparent and reliable price estimate.
## 2.Importing Libraries:
•	pandas: for handling and manipulating DataFrames.
•	ast: to convert strings containing python dictionaries to actual dictionaries.
•	glop and os: to search and retrieve the file paths from the file system.
•	numpy:  for handling numerical computation and missing values (e.g., np.NaN).
•	re: Used for regular expression to process text data (e.g., removing units like ‘CC’, ‘ Kmpl’)
•	matplotlib: used for creating statistical visualizations.
•	Seaborn: For creating a statistical visualization.
•	‘OneHotEncoder and LabelEncoder’: For encoding categorical variables into numerical formats for machine learning models.
•	StandardScaler: This transformed the data to have a mean of 0 and a standard deviation of 1.
•	Scipy.stats: provides statistical functions, probability distributions, statistical tests.
•	Pickle is used to load the trained machine learning model.
•	Streamlit is used to create an interactive web application.

## 3. Converting Unstructured Data into Structured Data
### Helper Functions
expand_top():
![image](https://github.com/user-attachments/assets/8bf58897-cb9f-48a7-8d85-f8fc1a39e32a)

This function takes a list of dictionaries (‘top_list’), extracts the value of each dictionary. And returns 
them as a list. This is used to extract top features from the ‘new_car_feature’ field in the dataset.
expand_data():
This function process a list of dictionaries (‘data_list’) where each dictionary has a ‘heading’ and a list of 
sub-features. The function extracts and returns the value from each sub-features list, organizing them 
under their respective headings.
![image](https://github.com/user-attachments/assets/ae3289bd-f344-4819-9cd4-8a1dab2b4692)
## Data Normalization:
### normalize_data():
This function expand and normalize the ‘new_car_feature’ field, which include both top_features and other detailed data.
![image](https://github.com/user-attachments/assets/535c3673-e0df-4dd7-b164-f12417e7518b)
## Generating Car Specification Data:
### Car_spec_data_generation():
This function process the car_overview and details to create a structured DataFrame
![image](https://github.com/user-attachments/assets/16f67704-93db-4606-8f6d-63522a9eb4e7)
## Combining DataFrame:
Combines ‘car_overview’,  ‘top_features’, ‘car_specification’ and ‘car_details’ DataFrames into new single DataFrame.
## Cleaning and Renaming Columns:
### Cleaned_final_df():
•	Cleans and rename columns to create a finalized DataFrame.
•	Remove duplicate columns: Drops columns that have same values in multiple instances.
•	Drop unwanted columns: Drops columns that are unnecessary for further analysis.
## Generating Structured Data From Excel File:
### generate_structured_excel_data:
Process multiple excel files, extracts relevant car information and structures the data into clean excel file
•	Reads each files and converts object string (stored as dictionaries) into usable data using ‘ast_literal_eval’.
•	Calls the respective helper functions to expand, normalize and clean the data.
•	Saved the cleaned data into new excel file. 
![image](https://github.com/user-attachments/assets/0d61a0ee-d3b4-48e3-a299-b7181ef27632)
![image](https://github.com/user-attachments/assets/ef616575-98ab-49da-88a0-dd8ccc02a9f7)
## 4. Preprocessing Structured Dataset:
### Retrieve all excel files:
![image](https://github.com/user-attachments/assets/501a6497-a042-4863-a1b9-dd45a54216b1)
This line uses globe module to retrieve all excel files ‘(*.xlsx)’ from the directory called ‘..cleaned_data/’ directory, each file in this folder represents car data for a specific city.
### Defining Cleaning Functions:
These functions are responsible for cleaning and processing specific columns in the dataset. Each functions targets one or more columns and applies transformations and returns the 
### cleaned DataFrame.
•	Clean third party insurance(df): Replaces inconsistent values in the ‘insurance_validity_period’ column with standardized terms like ‘Third_party’ and ‘zero_depreciation’.
•	Clean number of seats(df): Removes the word seats from the ‘number_of_seats’ column, converts the data to numeric format and fills missing values with the average number of seats.
•	Clean engine capacity(df): Removes the ‘CC’ from ‘engine_capacity’, converts the data to numbers, and fills the missing values with the average engine_capacity of the same car_model. If the average is still missing, it fills the missing value with the global average engine capacity.
•	Clean List_feature columns(df): Converts columns that contains list of features (like ‘comfort_and_convenience’) to strings and fills in missing values with the word ‘unknown’.
•	Clean Mileage(df): Removes units such as ‘kmpl’ and ‘km/kg’ from the ‘Mileage_(km/l)’ column and converts it to numeric values. It fills in missing mileage values with the overall average.
•	Clean Maximum power(value) and Clean maximum power(df): These functions clean the ‘Maximum_power’ column by converting different power units (e.g., PS, KW, bhp) into a consistent format(bhp) and then fills the missing values with the average power.
•	Clean torque(df): Cleans the torque column by removing units like ‘NM’, converting it to numeric format (bhp) and then fills the missing values with the average torque.
•	Clean wheel size(df): Removes the ‘R’ from the ‘wheel_size’ column and converts it to numbers, filling missing values with the average wheel_size.
•	Clean Battery Type(df): Fills the missing values in the Batterey_type column with the most frequent battery type.
•	Clean Kilometers Driven(df): Cleans the ‘Kilometers_driven’ column by removing commas and converting the values to a numeric format.
### Loop Through Each Excel Files: 
•	Iterates over each file in the ‘file_paths’ list.
•	Extracts the city name from the file name. for example, for ‘bangalore_cars.xlsx’, the city would be ‘Bangalore’.
•	Reads the excel file into a DataFrame(df) using ‘pd.read_excel()’.
•	Adds  a new column to the DataFrame called ‘city’ to keep track of which city each row of data belongs to.
### Apply Cleaning Functions:
•	This section applies each of the cleaning functions detailed earlier to clean different aspects of the data. For example, ‘ clean_third_party_insurance()’ ensures consistent values in the 
insurance column, while Clean_numner_of_seats()’ processes the seating capacity.
### Combined All Cleaned DataFrames:
Merges the cleaned datas into a large DataFrame and saved the final cleaned dataframe into a new excel file.(‘../preprocessed_cars/preprocessed_entire_data/.xlsx’)
![image](https://github.com/user-attachments/assets/8ad9ff4c-6c4c-4e0a-b17b-c1b2d109b379)
## 5. Encoding Preprocessed Dataset:
### Dataset Preprocessing:
The data was loaded from an excel file using ‘pandas’ and unnecessary columns like ‘top_features’, ‘comfor_and_convenience’, ‘Interior_features’, ‘Exterior_features’, ‘Safety_features’, ‘Entertainment_and_communication’.
### Handling Missing Values:
Missing values in numerical column were imputed using mean value with their respective column.
This ensures that the dataset remains incomplete without using bias.
### Outliers Found:
![image](https://github.com/user-attachments/assets/a4bc46e3-924b-4210-976f-2c61243cdd1e)
![image](https://github.com/user-attachments/assets/8847e12c-ecb1-4690-868d-28472a300587)
### Outliers  Removal:
To avoid the influence of extreme values, outliers were removed using Interquartile range (IQR) method on numerical columns. This helps in creating the more  reliable prediction model.
![image](https://github.com/user-attachments/assets/1ae731d0-fcf4-40ad-9ee2-3203d6158557)
### After removing outliers:
![image](https://github.com/user-attachments/assets/a91c29c5-de09-43a0-a2b2-617bfedc2db5)
### Encoding Categorical data:
Categorical variable such as ‘car_model’, ‘fuel_type’, ‘transmission_type’, ‘battery_type’, ‘city’ were one-hot encoded to convert them in numerical format for uses in machine Learning model. Label encoding were used for ‘Insurance_validity_period’ column.
### Feature Selection with RFE(Recursive Feature Elimination):
Using RFE with Linear Regression, Most significant features were selected for model training.
This reduced the dimensionality of the dataset and improving the model efficiency.
### Random Forest Regression Model:
A Random Forest Model trained on the reduced feature set. Random Forest was chosen due to its ability to handle large number of input variables and its resilience to overfitting.
### Model Evaluation:
The models were evaluated using metrics like Mean Absolute Error, Mean Squared Error, R-Squared value to access the accuracy of prediction.
•	Mean Absolute Error: This tells us, how far an average prediction were from the True values.
•	Root Mean Squared Error: This gives more weight to larger error, indicating how much the prediction deviate from the actual values.
•	R-Squared: This  tells us how much of the variation in the listed price is explained by the model.
![image](https://github.com/user-attachments/assets/de720444-d12d-4494-aaca-b8358de6fe3f)
### Pipeline Creation:
The machine Learning pipeline was designed to automate and streamline the data preprocessing and feature engineering and model training steps. This function ensure the consistency and reproducibility of the wolrkflow.
### Data Preprocessing: 
### Numerical feature processing:
•	Imputation of Missing Values: Missing numerical values were handling using a ‘SimpleImputer’, where the missing values were replaced with the mean of the respective column. This imputation strategy ensured that no data was lost due to missing values, while maintaining the integrity of the datasets.
•	Scaling Numerical Features: After imputation, numerical features were standardized using ‘StandardScaler’. This transformed the data to have a mean of 0 and a standard deviation of 1 which is essential for Random Forest Model to perform better, as it prevents features with larger ranges from dominating the model training process.
### Categorical Feature Processing:
•	One-Hot Encoding of Categorical Variable: Categorical column such as ‘car_model’, ‘fuel_type’, ‘transmission_type’ and others were transformed into binary features using ‘OneHotEncoder’, This method created separate column for each unique category( 1 means the value is presence, or 0 means not)
•	The ‘handle_unknown = ignore’ parameter ensured that any unseen categories during prediction did not cause errors by improving the robustness of pipeline.
### Model Selection and Training:
•	Random Forest Regressor: This model was used  as a machine learning algorithm. Random forest method that combines the output of multiple decision trees to improve prediction accuracy and control overfitting, in the pipeline, 100 decision trees were used (“n_estimators = 100”) and random state of 42 was set for reproducibility.
### Data Splitting:
•	The data was  split into training and testing sets using ‘train_test_split’. 80% of the data was used for training the model, while remaning 20% was reserved for testing. This split allowed for proper evaluation ofnthe model performance on unseen data, which provided insights into how well the model generalizes.
Model Training and Evaluation:
•	Once the pipeline was defined, it was fitted on the training data (‘x_train, y_train’), applying all preprocessing steps automatically before training the random forest regressor model.
•	The model was then evaluated on the test data (‘x_test, y_test’) using evaluation metrics such as Mean_Absolute_Error(MAE), Mean_Squared_Error(MSE), R_Squared(R2). This metrics helped in assessing the model’s performance in predicting the target variable( ‘Listed_price’).
### Pipeline Saved:
•	To ensure the pipeline could be reused in future, The trained pipeline was serialized and stored in pickle file.
5. Created an Interactive streamlit Application:
### Reading File:
•	It helps to locate the file related to the script’s location.
![image](https://github.com/user-attachments/assets/59245f37-54d7-4249-b585-58793d0f0bfc)
### Loading the Model:
•	This loads the pretrained random forest model which stored as a pickle file. This model is later used to predict car prices.
![image](https://github.com/user-attachments/assets/56a251c9-7134-4f5a-acae-6a45e8059c13)
### Loading the Data:
![image](https://github.com/user-attachments/assets/16a29652-f5e4-43d1-b78c-06e4fe64b8c9)
•	The code loads an excel file containing the dataset. This dataset is used to populate dropdown options in the sidebar. Allowing users to input car attributes based on real data.
### Sidebar Inputs for Car Features:
•	The app provides various dropdown and input numeric fields in the sidebar where it allows user to select the features of the cars to check price value.
•	Drop Down option: used for categorical features like ‘car_model’, ‘fuel_type’, ‘transmission_type’, ‘Battery_type’, ‘city’, ‘Insurance_validity_period’ are populated using unique 
### values from the DataFrame.
•	The number inputs allow user to specify numeric values for fields like ‘Mileage’, ‘Number_of_seats’, ‘Engine_capacity’, ‘Maximum_power’, ‘torque’, ‘kilometers_driven’, ‘wheel_size’, ‘Number_of_Owners’, ‘Model_year’ with pre defined ranges and default values.
### Creating a custom DataFrame:
•	Once the user inputs their preferred data, the app will display new DataFrame ‘custom_df’ on the app to organize all the selected features into the format which is expected by the model.
### Making Prediction:
•	When the ‘old_car_price_prediction_value’ button is clicked, ‘model.predict(custom_df)’ into the pre-trained Random Forest Model.
•	The predicted price then displayed using ‘st.success’ showing the estimated value of the car.

### Displaying Custom Input:
•	Before making the prediction, app displays the user input in the form of DataFrame.

![image](https://github.com/user-attachments/assets/44160683-735c-4d45-a960-e689a1ca9d1e)

## Conclusion:
The Car Dekho Project successfully implemented a Random Forest Regression model within a streamlined machine learning pipeline, achieving accurate price predictions for used cars. The thorough preprocessing of data, including feature encoding, scaling, and outlier removal, contributed to the model’s effectiveness. This project demonstrates the importance of data-driven approaches in the automotive industry and offers a practical tool for enhancing customer decision-making in the used car market.
