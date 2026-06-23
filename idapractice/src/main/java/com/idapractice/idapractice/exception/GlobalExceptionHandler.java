package com.idapractice.idapractice.exception;

import com.idapractice.idapractice.dto.ErrorResponseDTO;
import jakarta.servlet.http.HttpServletRequest;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.orm.ObjectOptimisticLockingFailureException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingRequestHeaderException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;

import java.util.stream.Collectors;

/**
 * Centralised exception → HTTP response mapping.
 *
 * All controllers are free of try/catch blocks.
 * Every failure path returns the same ErrorResponseDTO envelope.
 *
 * Status mapping:
 *  PaymentNotFoundException                → 404 Not Found
 *  InvalidPaymentOperationException        → 422 Unprocessable Entity
 *  IllegalArgumentException                → 400 Bad Request
 *  MethodArgumentNotValidException         → 400 Bad Request  (Bean Validation failures)
 *  HttpMessageNotReadableException         → 400 Bad Request  (malformed JSON)
 *  MethodArgumentTypeMismatchException     → 400 Bad Request  (wrong path variable type)
 *  MissingRequestHeaderException           → 400 Bad Request  (missing required header)
 *  ObjectOptimisticLockingFailureException → 409 Conflict     (@Version collision)
 *  DataIntegrityViolationException         → 409 Conflict     (DB unique constraint)
 *  Exception (catch-all)                   → 500 Internal Server Error
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    // ── 404 ──────────────────────────────────────────────────────────────────

    @ExceptionHandler(PaymentNotFoundException.class)
    public ResponseEntity<ErrorResponseDTO> handleNotFound(
            PaymentNotFoundException ex, HttpServletRequest req) {

        log.warn("Payment not found: {}", ex.getMessage());
        return build(HttpStatus.NOT_FOUND, "Not Found", ex.getMessage(), req);
    }

    // ── 400 ──────────────────────────────────────────────────────────────────

    /**
     * Handles @Valid failures on @RequestBody DTOs.
     * Collects every field error into one readable message so the caller
     * knows exactly what to fix in a single round-trip.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponseDTO> handleValidation(
            MethodArgumentNotValidException ex, HttpServletRequest req) {

        String details = ex.getBindingResult().getFieldErrors().stream()
                .map(fe -> fe.getField() + ": " + fe.getDefaultMessage())
                .collect(Collectors.joining("; "));

        log.warn("Validation failed: {}", details);
        return build(HttpStatus.BAD_REQUEST, "Validation Failed", details, req);
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponseDTO> handleUnreadableBody(
            HttpMessageNotReadableException ex, HttpServletRequest req) {

        log.warn("Malformed request body: {}", ex.getMessage());
        return build(HttpStatus.BAD_REQUEST, "Bad Request",
                "Request body is missing or contains malformed JSON", req);
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ErrorResponseDTO> handleTypeMismatch(
            MethodArgumentTypeMismatchException ex, HttpServletRequest req) {

        String msg = String.format("Parameter '%s' must be of type %s",
                ex.getName(),
                ex.getRequiredType() != null ? ex.getRequiredType().getSimpleName() : "unknown");
        log.warn("Type mismatch: {}", msg);
        return build(HttpStatus.BAD_REQUEST, "Bad Request", msg, req);
    }

    @ExceptionHandler(MissingRequestHeaderException.class)
    public ResponseEntity<ErrorResponseDTO> handleMissingHeader(
            MissingRequestHeaderException ex, HttpServletRequest req) {

        log.warn("Missing header: {}", ex.getHeaderName());
        return build(HttpStatus.BAD_REQUEST, "Bad Request",
                "Required header '" + ex.getHeaderName() + "' is missing", req);
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponseDTO> handleIllegalArgument(
            IllegalArgumentException ex, HttpServletRequest req) {

        log.warn("Illegal argument: {}", ex.getMessage());
        return build(HttpStatus.BAD_REQUEST, "Bad Request", ex.getMessage(), req);
    }

    // ── 409 ──────────────────────────────────────────────────────────────────

    /**
     * Optimistic lock collision — two concurrent updates raced on the same payment.
     * The losing request gets a 409; client should reload and retry.
     */
    @ExceptionHandler(ObjectOptimisticLockingFailureException.class)
    public ResponseEntity<ErrorResponseDTO> handleOptimisticLock(
            ObjectOptimisticLockingFailureException ex, HttpServletRequest req) {

        log.warn("Optimistic locking conflict on {}", ex.getPersistentClassName());
        return build(HttpStatus.CONFLICT, "Conflict",
                "Payment was modified by another request. Please reload and retry.", req);
    }

    /**
     * DB unique constraint violation (e.g. duplicate idempotency key with
     * a different payload, duplicate fromAccount+toAccount in some constraint).
     */
    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ErrorResponseDTO> handleDataIntegrity(
            DataIntegrityViolationException ex, HttpServletRequest req) {

        log.warn("Data integrity violation: {}", ex.getMessage());
        return build(HttpStatus.CONFLICT, "Conflict",
                "Request conflicts with existing data (duplicate key or constraint violation)", req);
    }

    // ── 422 ──────────────────────────────────────────────────────────────────

    @ExceptionHandler(InvalidPaymentOperationException.class)
    public ResponseEntity<ErrorResponseDTO> handleInvalidOperation(
            InvalidPaymentOperationException ex, HttpServletRequest req) {

        log.warn("Invalid payment operation: {}", ex.getMessage());
        return build(HttpStatus.UNPROCESSABLE_ENTITY, "Unprocessable Entity",
                ex.getMessage(), req);
    }

    
    @ExceptionHandler(AccountNotFoundException.class)
    public ResponseEntity<ErrorResponseDTO> handleInvalidAccountNumber(
            AccountNotFoundException ex, HttpServletRequest req) {

        log.warn("Invalid From/To Account: {}", ex.getMessage());
        return build(HttpStatus.UNPROCESSABLE_ENTITY, "Unprocessable Entity",
                ex.getMessage(), req);
    }

    // ── 500 ──────────────────────────────────────────────────────────────────

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponseDTO> handleAll(
            Exception ex, HttpServletRequest req) {

        // Full stack trace at ERROR level — this should page on-call
        log.error("Unhandled exception on {} {}", req.getMethod(), req.getRequestURI(), ex);
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "Internal Server Error",
                "An unexpected error occurred. Please contact support.", req);
    }

    // ── Helper ───────────────────────────────────────────────────────────────

    private ResponseEntity<ErrorResponseDTO> build(
            HttpStatus status, String error, String message, HttpServletRequest req) {

        return ResponseEntity.status(status)
                .body(new ErrorResponseDTO(status.value(), error, message, req.getRequestURI()));
    }
}
