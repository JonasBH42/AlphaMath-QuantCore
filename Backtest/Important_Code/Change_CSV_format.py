import csv
import sys

def change_csv_format(input_file, output_file, old_delimiter='-', new_delimiter=','):
   
    try:
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                reader = csv.reader(infile, delimiter=old_delimiter)
                writer = csv.writer(outfile, delimiter=new_delimiter)
                
                for row in reader:
                    writer.writerow(row)
        
        print(f"Successfully converted {input_file} to {output_file}")
        
    except FileNotFoundError:
        print(f"Error: File {input_file} not found")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
if __name__ == "__main__":
    if len(sys.argv) >= 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        change_csv_format(input_file, output_file)
    else:
        # Default example
        input_file = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/inventories.csv"
        output_file = "C:/Users/User/Desktop/AlphaMath-QuantCore/Backtest/Csvs/inventories_converted.csv"
        change_csv_format(input_file, output_file)