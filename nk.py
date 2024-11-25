import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, lit
from typing import List

def get_rows_without_nulls(session: snowpark.Session, table_name: str, limit: int = 50) -> snowpark.DataFrame:
    """
    Get rows from a Snowflake table where no columns contain NULL values.
    """
    df = session.table(table_name)
    columns: List[str] = df.columns
    
    condition = None
    for col_name in columns:
        if condition is None:
            condition = col(col_name).is_not_null()
        else:
            condition = condition & col(col_name).is_not_null()
    
    return df.filter(condition).limit(limit)

def main(session: snowpark.Session):
    """
    Main function to execute the non-null query
    """
    try:
        # Replace with your table name
        table_name = "YOUR_TABLE_NAME"
        
        # Get results
        result_df = get_rows_without_nulls(session, table_name)
        
        # Display results
        print(f"Found {result_df.count()} rows with no null values:")
        result_df.show()
        
        # Optionally convert to pandas
        # pandas_df = result_df.to_pandas()
        
        return result_df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Create Snowflake session
    session = snowpark.Session.builder.configs({
        "account": "your_account",
        "user": "your_user",
        "password": "your_password",
        "role": "your_role",
        "warehouse": "your_warehouse",
        "database": "your_database",
        "schema": "your_schema"
    }).create()
    
    try:
        main(session)
    finally:
        session.close()
