-- Score above threshold
SELECT
    *
FROM
    second_table
WHERE
    score >= 10
ORDER BY
    score DESC;
