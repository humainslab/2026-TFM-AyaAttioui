import pandas as pd

# Load the CSV file
df = pd.read_csv('data/blood-transfusion.csv')

# Remove the 'id' column if it exists
if 'id' in df.columns:
    df = df.drop('id', axis=1)
else:
    print("Column 'id' not found in dataset")

# Save the modified dataset
df.to_csv('data/blood-transfusion.csv', index=False)

