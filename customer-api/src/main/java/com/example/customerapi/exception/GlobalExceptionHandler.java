package com.example.customerapi.exception;
import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ErrorResponse> handleDuplicateSsn(
            DataIntegrityViolationException ex) {

        // Minimal & safe detection
        String message = ex.getMostSpecificCause().getMessage();

        if (message != null && message.toLowerCase().contains("customer_ssn")) {
            ErrorResponse error = new ErrorResponse(
                    "DUPLICATE_SSN",
                    "Customer with given SSN already exists"
            );
            return new ResponseEntity<>(error, HttpStatus.CONFLICT);
        }

        // Fallback for other DB integrity issues
        ErrorResponse error = new ErrorResponse(
                "DATA_INTEGRITY_ERROR",
                "Database constraint violation"
        );
        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    @ExceptionHandler(BusinessException.class)
    public ResponseEntity<ErrorResponse> handleBusinessException(BusinessException ex) {

        ErrorResponse error = new ErrorResponse(
                ex.getErrorCode(),
                ex.getMessage()
        );

        return ResponseEntity
                .status(HttpStatus.BAD_REQUEST)
                .body(error);
    }

}
