import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('ds_salaries.csv')

# Display the first few rows of the dataframe
df.head()

# Display summary statistics and info about the dataset
df.describe()
df.info()

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Dataset Missing Values Heatmap')
plt.show()


# List of numerical features you want to plot
numerical_features = ['salary', 'salary_in_usd', 'remote_ratio']

# Create histograms for numerical features
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[feature], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel(feature, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
# List of categorical features you want to plot
categorical_features = ['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']

# Create count plots for categorical features
for feature in categorical_features:
    plt.figure(figsize=(12, 8))
    chart = sns.countplot(y=feature, data=df, order=df[feature].value_counts().index)
    plt.title(f'Distribution of {feature}', fontsize=16)
    plt.xlabel('Frequency', fontsize=14)
    plt.ylabel(feature, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()


    # Set a theme for better aesthetics
sns.set_theme(style="whitegrid")

# Create an improved line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x='work_year', y='salary_in_usd', data=df, marker='o', color='blue', linewidth=2, markersize=8)

# Add a title and labels with a larger font size
plt.title('Salary Trends Over Work Years', fontsize=16, fontweight='bold')
plt.xlabel('Work Year', fontsize=14)
plt.ylabel('Salary (USD)', fontsize=14)

# Customize the ticks
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Remove the top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.show()


# Boxplot for Salary by Experience Level
plt.figure(figsize=(8, 6))  # Adjust size for better clarity
sns.boxplot(x='experience_level', y='salary_in_usd', data=df)

# Add a descriptive title and axis labels
plt.title('Salary Distribution by Experience Level', fontsize=16, fontweight='bold')
plt.xlabel('Experience Level', fontsize=14)
plt.ylabel('Salary (USD)', fontsize=14)

# Customize tick labels
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines for reference
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Remove spines for a cleaner look
sns.despine()

# Show the plot
plt.tight_layout()
plt.show()



# Improved boxplot for Salary by Job Title with adjusted usage
plt.figure(figsize=(12, 10))  # Increase height for better readability
sns.boxplot(y='job_title', x='salary_in_usd', data=df, color='skyblue')  # Use a single color

# Add a descriptive title and labels with larger font sizes
plt.title('Salary Distribution by Job Title', fontsize=16, fontweight='bold')
plt.xlabel('Salary (USD)', fontsize=14)
plt.ylabel('Job Title', fontsize=14)

# Customize tick labels for readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=10)

# Remove spines for a cleaner look
sns.despine()

# Show the plot
plt.tight_layout()  # Adjust layout to prevent text overlap
plt.show()



title_mapping = {
    'Data Scientist': [
        'Data Scientist', 'Lead Data Scientist', 'Applied Data Scientist', 'Principal Data Scientist'
    ],
    'Data Engineer': [
        'Data Engineer', 'Lead Data Engineer', 'Big Data Engineer', 'Principal Data Engineer', 'Cloud Data Engineer', 'Big Data Architect'
    ],
    'Machine Learning Specialist': [
        'Machine Learning Specialist', 'Machine Learning Engineer', 'Lead Machine Learning Engineer', 'Head of Machine Learning', 'ML Engineer', 'Machine Learning Developer', 'NLP Engineer', 'Machine Learning Scientist','Machine Learning Manager', 'Head of Machine Learning'
    ],
    'Data Analyst': [
        'Data Analyst', 'Business Data Analyst', 'Principal Data Analyst', 'Product Data Analyst', 'Finance Data Analyst', 'Lead Data Analyst'
    ],
    'Data Science Manager': [
        'Data Science Manager', 'Head of Data Science', 'Director of Data Science', 'Data Science Engineer', 'Data Analytics Manager', 'Data Analytics Lead', 'Machine Learning Manager', 'Head of Data'
    ],
    'Data Infrastructure': [
        'Data Infrastructure Engineer', 'Data Science Infrastructure Engineer', 'Data Engineering Manager'
    ],
    'Research and Development': [
        'Research Scientist', 'Data Science Consultant', '3D Computer Vision Researcher'
    ],
    'Data Architecture': [
        'Data Architect', 'Data Specialist', 'ETL Developer'
    ],
    'Computer Vision and AI': [
        'Computer Vision Engineer', 'Computer Vision Software Engineer', 'AI Scientist', 'Applied Machine Learning Scientist'
    ],
    'Analytics and Business Intelligence': [
        'Data Analytics Engineer', 'BI Data Analyst', 'Analytics Engineer', 'Staff Data Scientist', 'Marketing Data Analyst'
    ]
}

df['job_category'] = df['job_title'].replace({k: v for v, ks in title_mapping.items() for k in ks})
import seaborn as sns
import matplotlib.pyplot as plt

# Set the aesthetic style of the plots
sns.set(style='whitegrid')

# Create a boxplot for the new job categories
plt.figure(figsize=(12, 8))
sns.boxplot(x='salary_in_usd', y='job_category', data=df)
plt.title('Salary Distribution by Job Category', fontsize=16)
plt.xlabel('Salary (USD)', fontsize=14)
plt.ylabel('Job Category', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()



# Set a larger figure size for better visibility and readability
plt.figure(figsize=(14, 10))

# Create the boxplot with adjusted aesthetics
sns.boxplot(x='employee_residence', y='salary_in_usd', data=df,
            width=0.8)  # Adjust the width of the boxes

# Rotate x-axis labels for better visibility of country names
plt.xticks(rotation=45, fontsize=12, ha='right')  # Adjust rotation and alignment

# Add titles and labels with larger fonts for clarity
plt.title('Salary Distribution by Employee Residence', fontsize=20, fontweight='bold')
plt.xlabel('Employee Residence', fontsize=16)
plt.ylabel('Salary (USD)', fontsize=16)

# Enhance y-axis ticks for better granularity
plt.yticks(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Add grid lines for better analysis

# Improve layout to ensure no label cut-offs
plt.tight_layout()

# Remove the top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.show()


region_mapping = {
    'Americas': [
        'US', 'CA', 'MX',  # North America
        'BR', 'AR', 'CL', 'PE', 'CO', 'VE', 'EC',  # South America
        'HN'  # Central America
    ],
    'Europe': [
        'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'PL', 'CH', 'AT', 'BE', 'HU', 'PT', 'GR',
        'DK', 'RU', 'HR', 'BG', 'IQ', 'VN', 'UA', 'MT', 'RO', 'MD', 'SI', 'TR', 'RS', 'PR', 'LU',
        'CZ', 'DZ', 'TN', 'EE', 'LV', 'LT', 'IE','JE'
    ],
    'Asia': [
        'CN', 'JP', 'IN', 'SG', 'HK', 'KR', 'TW', 'TH', 'MY', 'PH','PK'
    ],
    'Australia and Oceania': [
        'AU', 'NZ'
    ],
    'Africa': [
        'ZA', 'NG', 'EG', 'KE', 'GH','BO'
    ],
    'Middle East': [
        'AE', 'SA', 'IL', 'QA', 'OM', 'KW', 'IR'
    ]
}

# Function to apply mapping
def map_residence_to_region(country):
    for region, countries in region_mapping.items():
        if country in countries:
            return region

# Apply the mapping
df['region'] = df['employee_residence'].apply(map_residence_to_region)

# Create a boxplot to visualize salary distributions by the mapped region
plt.figure(figsize=(14, 10))
sns.boxplot(x='region', y='salary_in_usd', data=df, width=0.8)  # Use 'region' instead of 'employee_residence'

# Rotate x-axis labels for better visibility of region names
plt.xticks(rotation=45, fontsize=12, ha='right')  # Adjust rotation and alignment

# Add titles and labels with larger fonts for clarity
plt.title('Salary Distribution by Region', fontsize=20, fontweight='bold')
plt.xlabel('Region', fontsize=16)
plt.ylabel('Salary (USD)', fontsize=16)

# Enhance y-axis ticks for better granularity
plt.yticks(fontsize=12)
plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Add grid lines for better analysis

# Improve layout to ensure no label cut-offs
plt.tight_layout()

# Remove the top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.show()


average_salary_by_residence = df.groupby('employee_residence')['salary_in_usd'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))

# Plot the average salary by employee residence using a bar chart
average_salary_by_residence = df.groupby('employee_residence')['salary_in_usd'].mean().sort_values()
average_salary_by_residence.plot(kind='bar', color='dodgerblue')

# Add a descriptive title and axis labels with larger fonts for clarity
plt.title('Average Salary by Employee Residence', fontsize=20, fontweight='bold')
plt.xlabel('Employee Residence', fontsize=16)
plt.ylabel('Average Salary (USD)', fontsize=16)

# Customize tick labels for better readability
plt.xticks(rotation=90, fontsize=10)  # Rotate labels for better readability of country names
plt.yticks(fontsize=12)

# Add grid lines for reference
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Improve layout to ensure no label cut-offs
plt.tight_layout()

# Remove the top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.show()



region_mapping = {
    'Americas': [
        'US', 'CA', 'MX',  # North America
        'BR', 'AR', 'CL', 'PE', 'CO', 'VE', 'EC',  # South America
        'HN'  # Central America
    ],
    'Europe': [
        'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'SE', 'NO', 'PL', 'CH', 'AT', 'BE', 'HU', 'PT', 'GR',
        'DK', 'RU', 'HR', 'BG', 'IQ', 'VN', 'UA', 'MT', 'RO', 'MD', 'SI', 'TR', 'RS', 'PR', 'LU',
        'CZ', 'DZ', 'TN', 'EE', 'LV', 'LT', 'IE','JE'
    ],
    'Asia': [
        'CN', 'JP', 'IN', 'SG', 'HK', 'KR', 'TW', 'TH', 'MY', 'PH','PK'
    ],
    'Australia and Oceania': [
        'AU', 'NZ'
    ],
    'Africa': [
        'ZA', 'NG', 'EG', 'KE', 'GH','BO'
    ],
    'Middle East': [
        'AE', 'SA', 'IL', 'QA', 'OM', 'KW', 'IR'
    ]
}

# Function to apply mapping
def map_residence_to_region(country):
    for region, countries in region_mapping.items():
        if country in countries:
            return region

# Apply the mapping
df['region'] = df['employee_residence'].apply(map_residence_to_region)
average_salary_by_residence = df.groupby('region')['salary_in_usd'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 8))

# Plot the average salary by employee residence using a bar chart
average_salary_by_residence = df.groupby('region')['salary_in_usd'].mean().sort_values()
average_salary_by_residence.plot(kind='bar', color='dodgerblue')

# Add a descriptive title and axis labels with larger fonts for clarity
plt.title('Average Salary by Employee Residence', fontsize=20, fontweight='bold')
plt.xlabel('Region', fontsize=16)
plt.ylabel('Average Salary (USD)', fontsize=16)

# Customize tick labels for better readability
plt.xticks(rotation=90, fontsize=10)  # Rotate labels for better readability of country names
plt.yticks(fontsize=12)

# Add grid lines for reference
plt.grid(axis='y', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)

# Improve layout to ensure no label cut-offs
plt.tight_layout()

# Remove the top and right spines for a cleaner look
sns.despine()

# Show the plot
plt.show()

# Enhanced scatterplot for Salary vs. Remote Ratio
plt.figure(figsize=(8, 6))
sns.scatterplot(x='remote_ratio', y='salary_in_usd', data=df, alpha=0.7, color='blue', s=50)

# Add a descriptive title and labels with larger font sizes
plt.title('Salary vs. Remote Ratio', fontsize=16, fontweight='bold')
plt.xlabel('Remote Ratio (%)', fontsize=14)
plt.ylabel('Salary (USD)', fontsize=14)

# Customize tick labels for better readability
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Add grid lines for reference
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)

# Remove spines for a cleaner look
sns.despine()

# Show the plot
plt.tight_layout()
plt.show()



# Now you can plot using this new 'company_size' column
plt.figure(figsize=(10, 6))
sns.boxplot(x='company_size', y='salary_in_usd', data=df)
plt.title('Salary Distribution by Company Size', fontsize=18, fontweight='bold')
plt.xlabel('Company Size', fontsize=14)
plt.ylabel('Salary (USD)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.despine()
plt.tight_layout()
plt.show()


correlation_matrix = df[['salary_in_usd','remote_ratio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

print(df.columns)

# Selecting the target and categorical features
target_variable = 'salary_in_usd'
categorical_features = ['work_year', 'experience_level', 'region', 'job_category', 'company_size']

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

# Apply OneHotEncoder to the categorical features
encoded_features = encoder.fit_transform(df[categorical_features])

# Create a DataFrame with the encoded features
encoded_columns = encoder.get_feature_names_out(categorical_features)
encoded_df = pd.DataFrame(encoded_features, columns=encoded_columns)

# Concatenate the target variable back to the encoded dataframe
encoded_df[target_variable] = df[target_variable]

# Calculate the correlation matrix for all encoded features plus the salary
correlation_matrix = encoded_df.corr()

# Filter the matrix to show only correlations with the salary
salary_correlations = correlation_matrix[[target_variable]].drop(target_variable)

# Visualize the correlation matrix
plt.figure(figsize=(8, 8))
sns.heatmap(salary_correlations, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix for Salary and Categorical Features')
plt.show()


# Selecting categorical columns to normalize
categorical_features = ['region', 'experience_level', 'job_category']

# Create the OneHotEncoder object
encoder = OneHotEncoder()  

# Create a column transformer to apply the encoder
column_transformer = ColumnTransformer(
    [("cat", encoder, categorical_features)],
    remainder='passthrough'  # This option allows us to keep the other columns in the DataFrame
)

# Fit and transform the data; it applies OneHotEncoding to the specified categorical features
encoded_data = column_transformer.fit_transform(df)

# Get new column names from the encoder
encoded_columns = column_transformer.named_transformers_['cat'].get_feature_names_out(categorical_features)

# Other columns that were not encoded
remaining_columns = [col for col in df.columns if col not in categorical_features]

# Full list of new column names
new_columns = list(encoded_columns) + remaining_columns

# Construct the new DataFrame
encoded_df = pd.DataFrame(encoded_data, columns=new_columns)

# Display the transformed DataFrame
print(encoded_df.head())
from sklearn.preprocessing import StandardScaler

# Assuming df is your DataFrame and salary_in_usd needs standardization
scaler = StandardScaler()
df['salary_in_usd_scaled'] = scaler.fit_transform(df[['salary_in_usd']])




from sklearn.model_selection import train_test_split

X = encoded_df.drop('salary_in_usd', axis=1)
y = encoded_df['salary_in_usd']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Print the shapes of the resulting sets to verify the split
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
