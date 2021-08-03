-- Create table users with enumerated countries
CREATE TABLE IF NOT EXISTS
    users(
        id INTEGER NOT NULL AUTO_INCREMENT,
        email VARCHAR(255) NOT NULL UNIQUE,
        name VARCHAR(255),
        PRIMARY KEY(id),
        country ENUM('US', 'CO', 'TN') NOT NULL DEFAULT 'US'
    );
