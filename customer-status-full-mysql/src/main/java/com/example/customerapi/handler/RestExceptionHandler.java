package com.example.customerapi.handler;

import com.example.customerapi.exception.CustomerStatusNotFoundException;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

@RestControllerAdvice
public class RestExceptionHandler {

    @ExceptionHandler(CustomerStatusNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleNotFound(CustomerStatusNotFoundException ex) {
        return ResponseEntity
                .status(404)
                .body(new ErrorResponse(404, ex.getMessage()));
    }

    public static class ErrorResponse {
        private int status;
        private String message;

        public ErrorResponse(int status, String message) {
            this.status = status;
            this.message = message;
        }

        public int getStatus() {
            return status;
        }

        public String getMessage() {
            return message;
        }
    }
}