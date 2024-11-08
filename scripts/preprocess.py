import pandas as pd

def load_datasets():

    try:
        # Define columns to keep for each dataset
        columns_config = {
            'services': ['Customer ID', 'Tenure in Months', 'Phone Service',
                         'Internet Service', 'Streaming', 'Monthly Charge', 'Total Charges'],
            'demographics': ['Customer ID', 'Age', 'Gender'],
            'location': ['Customer ID', 'City', 'State', 'Zip Code', 'Latitude', 'Longitude'],
            'status': ['Customer ID', 'Churn Value', 'Churn Category', 'Churn Reason']
        }

        # Load each dataset and select columns
        dfs = {}
        for name, columns in columns_config.items():
            df = pd.read_excel(f'./data/raw/{name}.xlsx')
            dfs[name] = df[columns]

        return dfs
    except Exception as e:
        print(f"Error loading datasets: {str(e)}")
        raise


def merge_and_sample_datasets(dfs, target_total=3738):
    try:
        # Merge datasets
        merged_df = dfs['services']
        for name in ['demographics', 'location', 'status']:
            merged_df = pd.merge(merged_df, dfs[name],
                                 on='Customer ID', how='inner')

        # Ensure Churn Value is numeric
        if merged_df['Churn Value'].dtype == 'object':
            merged_df['Churn Value'] = merged_df['Churn Value'].map({'No': 0, 'Yes': 1})

        # Split into churned and non-churned customers
        churned_df = merged_df[merged_df['Churn Value'] == 1]
        non_churned_df = merged_df[merged_df['Churn Value'] == 0]

        # Calculate how many non-churned customers to sample
        n_churned = len(churned_df)
        n_non_churned_to_sample = target_total - n_churned

        if n_non_churned_to_sample <= 0:
            print(f"Warning: There are {n_churned} churned customers, which exceeds target of {target_total}")
            return churned_df.sample(n=target_total, random_state=42)

        # Sample from non-churned customers
        sampled_non_churned = non_churned_df.sample(n=n_non_churned_to_sample, random_state=42)

        # Combine all churned with sampled non-churned
        final_df = pd.concat([churned_df, sampled_non_churned])

        # Shuffle the final dataset
        final_df = final_df.sample(frac=1, random_state=42)

        print(f"\nDataset composition:")
        print(f"Total rows: {len(final_df)}")
        print(f"Churned customers (positive cases): {len(churned_df)}")
        print(f"Non-churned customers (sampled): {len(sampled_non_churned)}")
        print(f"\nClass distribution:")
        print(final_df['Churn Value'].value_counts(normalize=True))

        return final_df

    except Exception as e:
        print(f"Error merging and sampling datasets: {str(e)}")
        raise


def clean_and_transform_data(df):

    try:
        # Handle missing values
        numerical_cols = df.select_dtypes(include=['float', 'int']).columns
        categorical_cols = df.select_dtypes(include=['object']).columns

        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        df[categorical_cols] = df[categorical_cols].fillna('Unknown')

        # Set data types
        dtype_dict = {
            'Customer ID': 'object',
            'Tenure in Months': 'float',
            'Phone Service': 'category',
            'Internet Service': 'category',
            'Streaming': 'category',
            'Monthly Charge': 'float',
            'Total Charges': 'float',
            'Age': 'float',
            'Gender': 'category',
            'City': 'category',
            'State': 'category',
            'Zip Code': 'float',
            'Latitude': 'float',
            'Longitude': 'float',
            'Churn Value': 'int',
            'Churn Category': 'category',
            'Churn Reason': 'category'
        }
        df = df.astype(dtype_dict)

        return df
    except Exception as e:
        print(f"Error cleaning and transforming data: {str(e)}")
        raise


def process_and_save_data(target_total=3738):
    """Main function to process and save the data."""
    try:
        # Load datasets
        dfs = load_datasets()

        # Merge datasets and sample
        merged_df = merge_and_sample_datasets(dfs, target_total)

        # Clean and transform data
        processed_df = clean_and_transform_data(merged_df)

        # Save processed data
        processed_df.to_parquet('./data/processed/merged.parquet', index=False)
        print("\nMerged dataset saved successfully")

    except Exception as e:
        print(f"Error in data processing pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    process_and_save_data(target_total=3738)