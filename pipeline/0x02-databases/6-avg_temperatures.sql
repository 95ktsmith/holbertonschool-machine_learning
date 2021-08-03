-- Average temperature by city
SELECT
    city,
    AVG(value) AS avg_temp
FROM
    temperatures
GROUP BY 
    city
ORDER BY
    avg_tmp DESC;
