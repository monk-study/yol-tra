SELECT *
FROM (
  SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
  FROM my_table
)
WHERE NOT EXISTS (
  SELECT 1 
  FROM (
    SELECT *, ROW_NUMBER() OVER (ORDER BY RANDOM()) AS rn
    FROM my_table
  ) t2
  WHERE t2.rn = subquery.rn AND EXISTS (
    SELECT 1 
    FROM FLATTEN(OBJECT_CONSTRUCT(*), keys, values) 
    WHERE value IS NULL
  )
)
LIMIT 50;
