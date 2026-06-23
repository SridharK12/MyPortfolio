-- V1__create_payments_table.sql
-- Managed by Flyway — DO NOT edit. Create a new V2__ file for schema changes.

CREATE TABLE IF NOT EXISTS payments (
    payment_id       BIGINT          NOT NULL AUTO_INCREMENT,
    from_account     VARCHAR(100)    NOT NULL,
    to_account       VARCHAR(100)    NOT NULL,
    amount           DECIMAL(19, 4)  NOT NULL,
    remarks          VARCHAR(500),
    -- Stored as VARCHAR so values are human-readable in the DB.
    -- Enum values: PENDING | APPROVED | REJECTED | CANCELLED
    status           VARCHAR(20)     NOT NULL DEFAULT 'PENDING',
    -- Caller-supplied deduplication token for safe client retries
    idempotency_key  VARCHAR(100)    UNIQUE,
    -- @Version field: incremented by JPA on every UPDATE to detect concurrent writes
    version          BIGINT          NOT NULL DEFAULT 0,
    created_at       DATETIME(6),
    updated_at       DATETIME(6),

    PRIMARY KEY (payment_id),

    -- Covering indexes for the two most common query patterns
    INDEX idx_payments_status         (status),
    INDEX idx_payments_from_account   (from_account),
    INDEX idx_payments_status_account (status, from_account)
);
