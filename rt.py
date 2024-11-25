import csv
import json
import sys
from typing import Dict, Union, List, Any

def try_parse_json(value: str) -> Any:
    """
    Attempt to parse a string as JSON.
    Returns the parsed JSON if successful, otherwise returns the original string.
    """
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value

def try_convert_numeric(value: str) -> Union[int, float, str]:
    """
    Attempt to convert a string to numeric type.
    Returns int/float if successful, otherwise returns the original string.
    """
    try:
        # Try converting to int
        return int(value)
    except ValueError:
        try:
            # Try converting to float
            return float(value)
        except ValueError:
            # Return original string if not numeric
            return value

def convert_row_to_json(csv_file: str, row_number: int, delimiter: str = ',') -> Dict[str, Any]:
    """
    Convert a specific row from a CSV file to JSON format, handling JSON columns.
    
    Args:
        csv_file (str): Path to the CSV file
        row_number (int): The row number to convert (1-based index)
        delimiter (str): CSV delimiter character (default: ',')
    
    Returns:
        dict: JSON representation of the specified row
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If row number is invalid
        IndexError: If row number is out of range
    """
    try:
        # Validate row number
        if row_number < 1:
            raise ValueError("Row number must be greater than 0")
            
        with open(csv_file, 'r', newline='', encoding='utf-8') as file:
            # Use csv.DictReader to handle quoting and escaping properly
            csv_reader = csv.DictReader(file, delimiter=delimiter)
            
            if not csv_reader.fieldnames:
                raise ValueError("CSV file is empty or has no headers")
            
            # Convert row number to 0-based index
            target_row_idx = row_number - 1
            
            # Skip to the target row
            for i, row in enumerate(csv_reader):
                if i == target_row_idx:
                    result = {}
                    # Process each column
                    for header, value in row.items():
                        if value is None:
                            result[header] = None
                            continue
                            
                        # First try to parse as JSON
                        parsed_value = try_parse_json(value)
                        
                        # If it wasn't JSON and is still a string, try converting to numeric
                        if isinstance(parsed_value, str):
                            parsed_value = try_convert_numeric(parsed_value)
                            
                        result[header] = parsed_value
                    return result
            
            raise IndexError(f"Row {row_number} not found in CSV file")
            
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file '{csv_file}' not found")
    except UnicodeDecodeError:
        raise ValueError("CSV file encoding error. Try opening the file with a different encoding.")

def main():
    """
    Main function to run the script from command line.
    Usage: python script.py <csv_file> <row_number> [delimiter]
    """
    if len(sys.argv) not in [3, 4]:
        print("Usage: python script.py <csv_file> <row_number> [delimiter]")
        sys.exit(1)
        
    try:
        csv_file = sys.argv[1]
        row_number = int(sys.argv[2])
        delimiter = sys.argv[3] if len(sys.argv) == 4 else ','
        
        result = convert_row_to_json(csv_file, row_number, delimiter)
        
        # Pretty print the JSON output
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
