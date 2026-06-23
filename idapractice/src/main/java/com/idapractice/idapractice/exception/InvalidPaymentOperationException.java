package com.idapractice.idapractice.exception;

/**
 * Thrown when a caller attempts an operation that violates payment lifecycle rules.
 *
 * Examples:
 *  - Updating an APPROVED or REJECTED payment
 *  - Cancelling (DELETE) a payment that is already APPROVED
 *  - Authorizing a payment that is not PENDING
 *
 * Maps to HTTP 422 Unprocessable Entity.
 */
public class InvalidPaymentOperationException extends RuntimeException {

    public InvalidPaymentOperationException(String message) {
        super(message);
    }
}
