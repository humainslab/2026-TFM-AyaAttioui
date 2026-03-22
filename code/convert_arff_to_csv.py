from scipy.io import arff
import pandas as pd
import os


def convert_arff_to_csv(input_path, output_path, files):
    """
    Convert ARFF files to CSV format.

    Args:
        input_path (str): Directory containing ARFF files
        output_path (str): Directory to save CSV files
        files (list): List of ARFF filenames
    """

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in files:
        try:

            print(f"Processing {file}...")

            file_path = os.path.join(input_path, file)
            data, meta = arff.loadarff(file_path)

            df = pd.DataFrame(data)
            
            data, meta = arff.loadarff(os.path.join(input_path, file))
            df = pd.DataFrame(data)

            # Convert byte strings to regular strings
            for col in df.select_dtypes([object]):
                df[col] = df[col].apply(lambda x: x.decode("utf-8"))

            csv_name = file.replace(".arff", ".csv")
            output_file = os.path.join(output_path, csv_name)
            
            df.to_csv(output_file, index=False)

            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

if __name__ == "__main__":
    files = ["usps_bin.arff", "cnae-9_bin.arff", "isolet_bin.arff"]

    convert_arff_to_csv(
        input_path="data_arff",
        output_path="data",
        files=files
    )