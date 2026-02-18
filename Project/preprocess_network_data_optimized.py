"""
Network Flow Data Preprocessing Script - Optimized for UNSW-NB15 / Similar Datasets
Extracts and preprocesses features from network flow CSV data for ML analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import argparse
import os


def load_dataset(filepath, nrows=None):
    """Load CSV dataset and display basic information."""
    print(f"Loading dataset from: {filepath}")
    if nrows:
        print(f"Loading first {nrows} rows for testing...")
        df = pd.read_csv(filepath, nrows=nrows, low_memory=False)
    else:
        df = pd.read_csv(filepath, low_memory=False)
    
    print(f"\n✅ Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nMissing values:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("No missing values!")
    
    return df


def extract_basic_features(df):
    """
    Extract basic network flow features.
    Maps dataset-specific columns to standard feature names.
    """
    print("\n🎯 Extracting basic network flow features...")
    
    features = pd.DataFrame()
    
    # Source and Destination IPs
    if 'saddr' in df.columns:
        features['Source_IP'] = df['saddr']
    elif 'src_ip' in df.columns:
        features['Source_IP'] = df['src_ip']
    
    if 'daddr' in df.columns:
        features['Destination_IP'] = df['daddr']
    elif 'dst_ip' in df.columns:
        features['Destination_IP'] = df['dst_ip']
    
    # Protocol
    if 'proto' in df.columns:
        features['Protocol'] = df['proto']
    elif 'protocol' in df.columns:
        features['Protocol'] = df['protocol']
    
    # Packet Count
    if 'pkts' in df.columns:
        features['Packet_Count'] = df['pkts']
    elif 'packet_count' in df.columns:
        features['Packet_Count'] = df['packet_count']
    
    # Duration
    if 'dur' in df.columns:
        features['Duration'] = df['dur']
    elif 'duration' in df.columns:
        features['Duration'] = df['duration']
    
    # Bytes Transferred
    if 'bytes' in df.columns:
        features['Bytes_Transferred'] = df['bytes']
    elif 'total_bytes' in df.columns:
        features['Bytes_Transferred'] = df['total_bytes']
    
    # Source and Destination Packets (if available)
    if 'spkts' in df.columns:
        features['Source_Packets'] = df['spkts']
    
    if 'dpkts' in df.columns:
        features['Destination_Packets'] = df['dpkts']
    
    # Source and Destination Bytes (if available)
    if 'sbytes' in df.columns:
        features['Source_Bytes'] = df['sbytes']
    
    if 'dbytes' in df.columns:
        features['Destination_Bytes'] = df['dbytes']
    
    # Rate features (if available)
    if 'rate' in df.columns:
        features['Rate'] = df['rate']
    
    if 'srate' in df.columns:
        features['Source_Rate'] = df['srate']
    
    if 'drate' in df.columns:
        features['Destination_Rate'] = df['drate']
    
    # State (connection state)
    if 'state' in df.columns:
        features['State'] = df['state']
    
    # Ports (if available)
    if 'sport' in df.columns:
        features['Source_Port'] = df['sport']
    
    if 'dport' in df.columns:
        features['Destination_Port'] = df['dport']
    
    # Label columns (for supervised learning)
    if 'attack' in df.columns:
        features['Label'] = df['attack']
    elif 'label' in df.columns:
        features['Label'] = df['label']
    
    if 'category' in df.columns:
        features['Attack_Category'] = df['category']
    
    if 'subcategory' in df.columns:
        features['Attack_Subcategory'] = df['subcategory']
    
    print(f"Extracted {len(features.columns)} features: {features.columns.tolist()}")
    
    return features


def create_derived_features(df):
    """Create additional derived features from existing ones."""
    print("\n🔧 Creating derived features...")
    
    # Bytes per packet
    if 'Bytes_Transferred' in df.columns and 'Packet_Count' in df.columns:
        df['Bytes_Per_Packet'] = df['Bytes_Transferred'] / (df['Packet_Count'] + 1)  # +1 to avoid division by zero
    
    # Packets per second
    if 'Packet_Count' in df.columns and 'Duration' in df.columns:
        df['Packets_Per_Second'] = df['Packet_Count'] / (df['Duration'] + 0.001)  # +0.001 to avoid division by zero
    
    # Bytes per second (if not already present as Rate)
    if 'Bytes_Transferred' in df.columns and 'Duration' in df.columns and 'Rate' not in df.columns:
        df['Bytes_Per_Second'] = df['Bytes_Transferred'] / (df['Duration'] + 0.001)
    
    # Source to destination packet ratio
    if 'Source_Packets' in df.columns and 'Destination_Packets' in df.columns:
        df['Src_Dst_Packet_Ratio'] = df['Source_Packets'] / (df['Destination_Packets'] + 1)
    
    # Source to destination byte ratio
    if 'Source_Bytes' in df.columns and 'Destination_Bytes' in df.columns:
        df['Src_Dst_Byte_Ratio'] = df['Source_Bytes'] / (df['Destination_Bytes'] + 1)
    
    print(f"Total features after derivation: {len(df.columns)}")
    
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


def encode_categorical_features(df, encode_ips=True, encode_ports=False):
    """Encode categorical features for ML."""
    print("\n🔢 Encoding categorical features...")
    
    # Store label columns separately (don't encode these)
    label_cols = ['Label', 'Attack_Category', 'Attack_Subcategory']
    labels_df = df[label_cols].copy() if any(col in df.columns for col in label_cols) else None
    
    # Drop label columns temporarily
    df_to_encode = df.drop(columns=[col for col in label_cols if col in df.columns], errors='ignore')
    
    # One-hot encode Protocol
    if 'Protocol' in df_to_encode.columns:
        print("One-hot encoding Protocol...")
        df_to_encode = pd.get_dummies(df_to_encode, columns=['Protocol'], prefix='Protocol')
        protocol_cols = [col for col in df_to_encode.columns if col.startswith('Protocol_')]
        print(f"Protocol columns created: {protocol_cols}")
    
    # One-hot encode State
    if 'State' in df_to_encode.columns:
        print("One-hot encoding State...")
        df_to_encode = pd.get_dummies(df_to_encode, columns=['State'], prefix='State')
        state_cols = [col for col in df_to_encode.columns if col.startswith('State_')]
        print(f"State columns created: {state_cols}")
    
    # Label encode IP addresses
    if encode_ips:
        le_src = LabelEncoder()
        le_dst = LabelEncoder()
        
        if 'Source_IP' in df_to_encode.columns:
            print("Label encoding Source_IP...")
            df_to_encode['Source_IP'] = le_src.fit_transform(df_to_encode['Source_IP'].astype(str))
        
        if 'Destination_IP' in df_to_encode.columns:
            print("Label encoding Destination_IP...")
            df_to_encode['Destination_IP'] = le_dst.fit_transform(df_to_encode['Destination_IP'].astype(str))
    
    # Optionally encode ports
    if encode_ports:
        if 'Source_Port' in df_to_encode.columns:
            print("Label encoding Source_Port...")
            le_sport = LabelEncoder()
            df_to_encode['Source_Port'] = le_sport.fit_transform(df_to_encode['Source_Port'].astype(str))
        
        if 'Destination_Port' in df_to_encode.columns:
            print("Label encoding Destination_Port...")
            le_dport = LabelEncoder()
            df_to_encode['Destination_Port'] = le_dport.fit_transform(df_to_encode['Destination_Port'].astype(str))
    else:
        # Drop port columns if not encoding
        df_to_encode = df_to_encode.drop(columns=['Source_Port', 'Destination_Port'], errors='ignore')
    
    # Add labels back
    if labels_df is not None:
        for col in labels_df.columns:
            if col in labels_df:
                df_to_encode[col] = labels_df[col]
    
    return df_to_encode


def normalize_features(df, exclude_cols=None):
    """Normalize numerical features using StandardScaler."""
    print("\n📊 Normalizing numerical features...")
    
    if exclude_cols is None:
        exclude_cols = ['Label', 'Attack_Category', 'Attack_Subcategory']
    
    # Identify numeric columns to normalize
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
    
    if cols_to_normalize:
        print(f"Normalizing {len(cols_to_normalize)} columns...")
        scaler = StandardScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        print("✅ Normalization complete!")
    else:
        print("⚠️ No columns to normalize.")
    
    return df


def save_preprocessed_data(df, output_path):
    """Save preprocessed data to CSV."""
    print(f"\n💾 Saving preprocessed data to: {output_path}")
    df.to_csv(output_path, index=False)
    print(f"✅ Saved successfully! Shape: {df.shape}")
    print(f"\nFinal columns ({len(df.columns)}): {df.columns.tolist()}")
    print(f"\nSample of preprocessed data:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())
    
    # Show label distribution if available
    if 'Label' in df.columns:
        print(f"\nLabel distribution:")
        print(df['Label'].value_counts())
    
    if 'Attack_Category' in df.columns:
        print(f"\nAttack category distribution:")
        print(df['Attack_Category'].value_counts())


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(description='Preprocess network flow data')
    parser.add_argument('input_file', help='Input CSV file path')
    parser.add_argument('--output', '-o', default='preprocessed_features.csv',
                        help='Output CSV file path (default: preprocessed_features.csv)')
    parser.add_argument('--nrows', type=int, default=None,
                        help='Number of rows to load (for testing)')
    parser.add_argument('--no-encode-ips', action='store_true',
                        help='Skip IP address encoding')
    parser.add_argument('--encode-ports', action='store_true',
                        help='Encode port numbers (default: drop them)')
    parser.add_argument('--normalize', action='store_true',
                        help='Normalize numerical features')
    parser.add_argument('--no-derived', action='store_true',
                        help='Skip derived feature creation')
    
    args = parser.parse_args()
    
    # Verify input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        return
    
    print("=" * 80)
    print("NETWORK FLOW DATA PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Load dataset
    df = load_dataset(args.input_file, nrows=args.nrows)
    
    # Step 2: Extract basic features
    features = extract_basic_features(df)
    
    # Step 3: Create derived features
    if not args.no_derived:
        features = create_derived_features(features)
    
    # Step 4: Handle missing values
    features = handle_missing_values(features)
    
    # Step 5: Encode categorical features
    features = encode_categorical_features(
        features,
        encode_ips=not args.no_encode_ips,
        encode_ports=args.encode_ports
    )
    
    # Step 6: Normalize features (optional)
    if args.normalize:
        features = normalize_features(features)
    
    # Step 7: Save preprocessed data
    save_preprocessed_data(features, args.output)
    
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
