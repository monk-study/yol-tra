import snowflake.snowpark as snowpark
from snowflake.snowpark.functions import col, lit
from typing import List

def get_rows_without_nulls(session: snowpark.Session, table_name: str, limit: int = 50) -> snowpark.DataFrame:
    """
    Get rows from a Snowflake table where no columns contain NULL values.
    
    Args:
        session: Snowpark session
        table_name: Name of the table to query
        limit: Number of rows to return (default 50)
        
    Returns:
        DataFrame containing rows with no NULL values
    """
    # Get the DataFrame
    df = session.table(table_name)
    
    # Get all column names
    columns: List[str] = df.columns
    
    # Create a condition that checks if all columns are not null
    condition = None
    for col_name in columns:
        if condition is None:
            condition = col(col_name).is_not_null()
        else:
            condition = condition & col(col_name).is_not_null()
    
    # Apply the condition and get the results
    result_df = df.filter(condition).limit(limit)
    
    return result_df

# Example usage:
"""
# First create your session
session = snowpark.Session.builder.configs({
    "account": "your_account",
    "user": "your_user",
    "password": "your_password",
    "role": "your_role",
    "warehouse": "your_warehouse",
    "database": "your_database",
    "schema": "your_schema"
}).create()

# Then use the function
result = get_rows_without_nulls(session, "YOUR_TABLE_NAME")

# View results
result.show()

# Or convert to pandas if needed
pandas_df = result.to_pandas()
"""
