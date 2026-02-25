CREATE TABLE IF NOT EXISTS customer_status (
    customer_id BIGINT NOT NULL,
    status VARCHAR(100) NOT NULL,
    modified_by VARCHAR(100),
    modification_date DATETIME,
    PRIMARY KEY (customer_id)
);
