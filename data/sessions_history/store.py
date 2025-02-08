import os
import json
import argparse
import pandas as pd
from google.cloud import storage, bigquery
from google.oauth2 import service_account
import streamlit as st
from datetime import datetime


def load_json_from_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as file:
        return json.load(file)


def json_files_to_dataframe(folder_path):
    """Convert JSON files in a folder to a pandas DataFrame with the required structure."""
    data = [
        load_json_from_file(os.path.join(folder_path, file_name))
        for file_name in os.listdir(folder_path)
        if file_name.endswith('.json')
    ]

    # Flatten the nested JSON structure into a DataFrame
    records = []
    for entry_list in data:  # entry_list is a list of dictionaries
        for entry in entry_list:  # Iterate through each dictionary in the list
            record = {
                'session_id': entry.get('session_id'),
                'app_version': entry.get('app_version'),
                'location': entry.get('location', []),  # Ensure location is always a list, default to empty list if missing
                'country': entry.get('country'),
                'timestamp': entry.get('timestamp'),
                'medical_class': entry.get('medical_class'),
                'severity': entry.get('severity'),
                'queries': entry.get('queries', []),  # Default to empty list if missing
                'responses': entry.get('responses', []),  # Default to empty list if missing
                'response_times': entry.get('response_times', []),  # Default to empty list if missing
                'hospital_name': entry.get('hospital', {}).get('name', 'Unknown'),  # Safely access nested keys
                'hospital_gmaps_link': entry.get('hospital', {}).get('gmaps_link', 'N/A'),  # Safely access nested keys
                'youtube_video_title': entry.get('youtube_video', {}).get('title', 'N/A'),  # Safely access nested keys
                'youtube_video_link': entry.get('youtube_video', {}).get('link', 'N/A')  # Safely access nested keys
            }
            records.append(record)

    # Create DataFrame
    df = pd.DataFrame(records)

    # Convert 'timestamp' to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')  # 'errors=coerce' will convert invalid timestamps to NaT

    # Ensure 'location' is a list (it should already be, but check in case)
    df['location'] = df['location'].apply(lambda x: x if isinstance(x, list) else [])

    # Ensure 'queries', 'responses', and 'response_times' are lists
    df['queries'] = df['queries'].apply(lambda x: x if isinstance(x, list) else [])
    df['responses'] = df['responses'].apply(lambda x: x if isinstance(x, list) else [])
    df['response_times'] = df['response_times'].apply(lambda x: x if isinstance(x, list) else [])

    # Select only the required columns
    required_columns = [
        'session_id', 'app_version', 'location', 'country', 'timestamp', 'medical_class', 'severity', 'queries', 'responses',
        'response_times', 'hospital_name', 'hospital_gmaps_link', 'youtube_video_title', 'youtube_video_link'
    ]
    df = df[required_columns]
    
    return df


def get_existing_session_ids(bigquery_client, table_ref):
    """Fetch existing session_ids from a BigQuery table."""
    existing_rows = bigquery_client.list_rows(table_ref, selected_fields=[bigquery.SchemaField('session_id', 'STRING')])
    return {row['session_id'] for row in existing_rows}


def create_table_if_not_exists(bigquery_client, dataset_id, table_id, df):
    """Create a BigQuery table if it does not exist with partitioning and clustering."""
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)

    try:
        # Try to get the table schema
        table = bigquery_client.get_table(table_ref)
        print(f"Table '{table_id}' already exists.")
    except bigquery.exceptions.NotFound:
        # If table doesn't exist, create it using the DataFrame schema
        schema = [
            bigquery.SchemaField(name, "STRING" if df[name].dtype == 'object' else "FLOAT64" if df[name].dtype == 'float64' else "TIMESTAMP")
            for name in df.columns
        ]
        
        # Define the partitioning and clustering
        table = bigquery.Client().dataset(dataset_id).table(table_id, schema=schema)
        table.schema = schema
        table.time_partitioning = bigquery.TimePartitioning(field="timestamp", type_="DAY")  # Partition by the date of timestamp
        table.clustering_fields = ["medical_class", "severity"]  # Cluster by medical_class and severity
        
        bigquery_client.create_table(table)
        print(f"Table '{table_id}' created successfully.")


def json_files_to_bigquery(bucket_name, dataset_id, table_id, storage_client, bigquery_client):
    """Load new JSON data from GCS to BigQuery, skipping existing session_ids."""
    bucket = storage_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs()
    table_ref = bigquery_client.dataset(dataset_id).table(table_id)

    # Get the set of existing session_ids from BigQuery
    existing_session_ids = get_existing_session_ids(bigquery_client, table_ref)

    # Process and filter new data
    new_data = [
        json.loads(blob.download_as_string())
        for blob in blobs
        if blob.name.endswith('.json') and json.loads(blob.download_as_string())['session_id'] not in existing_session_ids
    ]

    if new_data:
        # Flatten the list of dictionaries into a DataFrame
        df = pd.json_normalize(new_data, sep='_')

        # Reformat the dataframe as required
        df = json_files_to_dataframe(df)

        # Create table if not exists
        create_table_if_not_exists(bigquery_client, dataset_id, table_id, df)

        # Append the new data to BigQuery without overwriting the existing table
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
        bigquery_client.load_table_from_dataframe(df, table_ref, job_config=job_config).result()

        print(f"Data successfully loaded into BigQuery table: {table_id}")


def process_local_json(folder_path):
    """Process JSON files from a local folder and store them in memory as a list of lists."""
    df = json_files_to_dataframe(folder_path)

    # Save the data to CSV as a list of lists with headers
    csv_file_path = os.path.join(folder_path, "sessions_history.csv")
    df.to_csv(csv_file_path, index=False, header=True)  # Ensure headers are included
    print(f"Data saved to CSV at: {csv_file_path}")
    
    return df  # Return the DataFrame if needed


def process_gcs_to_bigquery(bucket_name, dataset_id, table_id, service_account_path):
    """Process JSON files from GCS to BigQuery."""
    credentials = service_account.Credentials.from_service_account_file(service_account_path)
    storage_client = storage.Client(credentials=credentials)
    bigquery_client = bigquery.Client(credentials=credentials)
    
    json_files_to_bigquery(bucket_name, dataset_id, table_id, storage_client, bigquery_client)


def main():
    """
    Main function to process JSON files based on the specified modality.
    
    Modality 1: Convert JSON files from a local folder to a pandas DataFrame and store them in a CSV.
    Modality 2: Convert JSON files from a GCS bucket to a BigQuery table.
    """
    parser = argparse.ArgumentParser(description='Process JSON files.')
    parser.add_argument('modality', type=int, choices=[1, 2], help='1 for local path, 2 for BigQuery')
    args = parser.parse_args()

    # Local path mode: process and display in Streamlit
    if args.modality == 1:
        folder_path = '.'
        process_local_json(folder_path)

    # BigQuery mode: load data from GCS to BigQuery
    elif args.modality == 2:
        bucket_name = st.secrets["GCP"]["BUCKET_NAME"]
        dataset_id = st.secrets["GCP"]["BIGQUERY_DATASET"]
        table_id = st.secrets["GCP"]["BIGQUERY_TABLE"]
        service_account_path = st.secrets["GCP"]["SERVICE_ACCOUNT_KEY"]
        
        process_gcs_to_bigquery(bucket_name, dataset_id, table_id, service_account_path)


if __name__ == "__main__":
    main()
