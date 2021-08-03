-- Creates a prodedure that computes and stores a student's average score
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN user_id INT
)

BEGIN
    UPDATE users
        SET average_score = (
            SELECT
                AVG(score)
            FROM
                corrections
            WHERE
                corrections.user_id = user_id
        )
        WHERE
            ID = user_id;
END $$

DELIMITER;
