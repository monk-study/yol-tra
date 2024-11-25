-- Define the table name variable
SET my_table = 'employee';

SELECT *
FROM (
  SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
  FROM @{my_table}
)
WHERE NOT EXISTS (
  SELECT 1
  FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
    FROM @{my_table}  
  ) t2
  WHERE t2.rn = subquery.rn
  AND EXISTS (
    SELECT 1 
    FROM @{my_table} t3
    WHERE t3.ROW_NUMBER() = t2.rn
    AND OBJECT_AGG(t3.*, ',') LIKE '%,NULL,%'
  )
)
LIMIT 50;
