-- V2__create_payment_outbox_table.sql

CREATE TABLE IF NOT EXISTS payment_outbox (
    id              BIGINT          NOT NULL AUTO_INCREMENT,
    payment_id      BIGINT          NOT NULL,
    topic           VARCHAR(100)    NOT NULL,
    -- Kafka message key (paymentId as string) — ensures per-payment ordering
    message_key     VARCHAR(50)     NOT NULL,
    -- Serialised JSON of PaymentResponseDTO
    payload         TEXT            NOT NULL,
    -- False until PaymentOutboxPublisher successfully delivers to Kafka
    published       BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      DATETIME(6),
    published_at    DATETIME(6),

    PRIMARY KEY (id),

    -- The publisher queries by published=false ordered by created_at.
    -- This index covers that query entirely.
    INDEX idx_outbox_pending     (published, created_at),
    INDEX idx_outbox_payment_id  (payment_id)
);
