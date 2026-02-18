"""
Network Flow Data Preprocessing Script
Extracts and preprocesses features from network flow CSV data for ML analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import argparse
import os
from datetime import datetime


def load_dataset(filepath):
    """Load CSV dataset and display basic information."""
    print(f"Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    print(df.isnull().sum())
    return df


def map_column_names(df):
    """
    Intelligently map dataset columns to standard names.
    Handles common variations in column naming.
    """
    # Common column name mappings (case-insensitive)
    column_mapping = {
        # Source IP variations
        'src_ip': 'Source_IP',
        'source_ip': 'Source_IP',
        'srcip': 'Source_IP',
        'src ip': 'Source_IP',
        'source ip': 'Source_IP',
        
        # Destination IP variations
        'dst_ip': 'Destination_IP',
        'dest_ip': 'Destination_IP',
        'destination_ip': 'Destination_IP',
        'dstip': 'Destination_IP',
        'dst ip': 'Destination_IP',
        'destination ip': 'Destination_IP',
        
        # Protocol variations
        'protocol': 'Protocol',
        'proto': 'Protocol',
        
        # Packet count variations
        'packet_count': 'Packet_Count',
        'packets': 'Packet_Count',
        'pkt_count': 'Packet_Count',
        'num_packets': 'Packet_Count',
        'total_packets': 'Packet_Count',
        
        # Bytes variations
        'bytes': 'Bytes_Transferred',
        'total_bytes': 'Bytes_Transferred',
        'byte_count': 'Bytes_Transferred',
        'bytes_transferred': 'Bytes_Transferred',
        
        # Duration variations
        'duration': 'Duration',
        'flow_duration': 'Duration',
        
        # Time variations
        'start_time': 'Start_Time',
        'starttime': 'Start_Time',
        'end_time': 'End_Time',
        'endtime': 'End_Time',
        'timestamp': 'Timestamp',
    }
    
    # Create a case-insensitive mapping
    df_columns_lower = {col.lower().strip(): col for col in df.columns}
    
    rename_dict = {}
    for old_name, new_name in column_mapping.items():
        if old_name in df_columns_lower:
            original_col = df_columns_lower[old_name]
            rename_dict[original_col] = new_name
    
    if rename_dict:
        print(f"\n📝 Renaming columns: {rename_dict}")
        df = df.rename(columns=rename_dict)
    
    return df


def extract_protocol(df):
    """Extract and encode protocol information."""
    if 'Protocol' not in df.columns:
        print("⚠️ Warning: Protocol column not found. Skipping protocol extraction.")
        return df
    
    # Check if protocol is numeric
    if pd.api.types.is_numeric_dtype(df['Protocol']):
        print("\n🔄 Converting numeric protocol codes to names...")
        protocol_map = {
            0: 'HOPOPT',
            1: 'ICMP',
            2: 'IGMP',
            6: 'TCP',
            17: 'UDP',
            41: 'IPv6',
            47: 'GRE',
            50: 'ESP',
            51: 'AH',
            58: 'ICMPv6',
            89: 'OSPF',
            132: 'SCTP',
        }
        df['Protocol'] = df['Protocol'].map(protocol_map).fillna('OTHER')
    
    print(f"Protocol distribution:\n{df['Protocol'].value_counts()}")
    return df


def calculate_duration(df):
    """Calculate flow duration from start and end times."""
    if 'Duration' in df.columns:
        print("\n✅ Duration column already exists.")
        return df
    
    if 'Start_Time' in df.columns and 'End_Time' in df.columns:
        print("\n⏱️ Calculating duration from start and end times...")
        try:
            df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
            df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
            df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds()
            print(f"Duration statistics:\n{df['Duration'].describe()}")
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate duration: {e}")
    else:
        print("⚠️ Warning: No time columns found to calculate duration.")
    
    return df


def aggregate_flow_features(df):
    """
    Aggregate packet-level data to flow-level if needed.
    Groups by source IP, destination IP, and protocol.
    """
    # Check if data is already flow-level or packet-level
    if 'Packet_Count' not in df.columns:
        print("\n📊 Aggregating packet-level data to flow-level...")
        
        # Identify grouping columns
        group_cols = []
        if 'Source_IP' in df.columns:
            group_cols.append('Source_IP')
        if 'Destination_IP' in df.columns:
            group_cols.append('Destination_IP')
        if 'Protocol' in df.columns:
            group_cols.append('Protocol')
        
        if not group_cols:
            print("⚠️ Warning: Cannot aggregate - missing grouping columns.")
            return df
        
        # Aggregate
        agg_dict = {}
        
        if 'Bytes_Transferred' in df.columns:
            agg_dict['Bytes_Transferred'] = 'sum'
        
        if 'Duration' in df.columns:
            agg_dict['Duration'] = 'max'
        
        # Count packets per flow
        agg_dict['Packet_Count'] = 'size'
        
        df = df.groupby(group_cols, as_index=False).agg(agg_dict)
        print(f"Aggregated to {len(df)} flows.")
    
    return df


def select_features(df):
    """Select and organize final feature set."""
    print("\n🎯 Selecting final features...")
    
    # Define desired features in order
    desired_features = [
        'Source_IP',
        'Destination_IP',
        'Protocol',
        'Packet_Count',
        'Duration',
        'Bytes_Transferred'
    ]
    
    # Select only available features
    available_features = [col for col in desired_features if col in df.columns]
    
    if not available_features:
        print("⚠️ Warning: No standard features found in dataset!")
        print("Available columns:", df.columns.tolist())
        return df
    
    print(f"Selected features: {available_features}")
    features_df = df[available_features].copy()
    
    return features_df


def encode_categorical_features(df):
    """Encode categorical features for ML."""
    print("\n🔢 Encoding categorical features...")
    
    # One-hot encode Protocol
    if 'Protocol' in df.columns:
        print("One-hot encoding Protocol...")
        df = pd.get_dummies(df, columns=['Protocol'], prefix='Protocol')
        print(f"Protocol columns created: {[col for col in df.columns if col.startswith('Protocol_')]}")
    
    # Label encode IP addresses
    le_src = LabelEncoder()
    le_dst = LabelEncoder()
    
    if 'Source_IP' in df.columns:
        print("Label encoding Source_IP...")
        df['Source_IP'] = le_src.fit_transform(df['Source_IP'].astype(str))
    
    if 'Destination_IP' in df.columns:
        print("Label encoding Destination_IP...")
        df['Destination_IP'] = le_dst.fit_transform(df['Destination_IP'].astype(str))
    
    return df


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    print("\n🔧 Handling missing values...")
    
    missing_before = df.isnull().sum().sum()
    print(f"Total missing values before: {missing_before}")
    
    if missing_before > 0:
        print("Missing values per column:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
        
        # Fill numeric columns with 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Fill categorical columns with 'UNKNOWN'
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna('UNKNOWN')
    
    missing_after = df.isnull().sum().sum()
    print(f"Total missing values after: {missing_after}")
    
    return df


def save_preprocessed_data(df, output_path):
    """Save preprocessed data to CSV."""
    print(f"\n💾 Saving preprocessed data to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved successfully! Shape: {df.shape}")
    print(f"\nFinal columns: {df.columns.tolist()}")
    print(f"\nSample of preprocessed data:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess network flow data')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='preprocessed_features.csv',
                        help='Output CSV file path (default: preprocessed_features.csv)')
    parser.add_argument('--no-aggregate', action='store_true',
                        help='Skip flow aggregation step')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        return
    
    print("=" * 80)
    print("NETWORK FLOW DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load dataset
    df = load_dataset(args.input_file)
    
    # Step 2: Map column names
    df = map_column_names(df)
    
    # Step 3: Extract protocol information
    df = extract_protocol(df)
    
    # Step 4: Calculate duration
    df = calculate_duration(df)
    
    # Step 5: Aggregate to flow-level (if needed)
    if not args.no_aggregate:
        df = aggregate_flow_features(df)
    
    # Step 6: Select features
    df = select_features(df)
    
    # Step 7: Handle missing values
    df = handle_missing_values(df)
    
    # Step 8: Encode categorical features
    df = encode_categorical_features(df)
    
    # Step 9: Save preprocessed data
    save_preprocessed_data(df, args.output)
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
