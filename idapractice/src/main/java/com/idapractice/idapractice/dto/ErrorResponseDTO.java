package com.idapractice.idapractice.dto;

import com.fasterxml.jackson.annotation.JsonFormat;

import java.time.LocalDateTime;

/**
 * Uniform error body for every non-2xx response.
 *
 * {
 *   "status"    : 404,
 *   "error"     : "Not Found",
 *   "message"   : "Payment not found with id: 99",
 *   "path"      : "/v1/payments/99",
 *   "timestamp" : "2026-06-17T10:30:00"
 * }
 */
public class ErrorResponseDTO {

    private int status;
    private String error;
    private String message;
    private String path;

    @JsonFormat(pattern = "yyyy-MM-dd'T'HH:mm:ss")
    private LocalDateTime timestamp;

    public ErrorResponseDTO() {
        this.timestamp = LocalDateTime.now();
    }

    public ErrorResponseDTO(int status, String error, String message, String path) {
        this.status    = status;
        this.error     = error;
        this.message   = message;
        this.path      = path;
        this.timestamp = LocalDateTime.now();
    }

    public int getStatus()                      { return status; }
    public void setStatus(int v)                { this.status = v; }

    public String getError()                    { return error; }
    public void setError(String v)              { this.error = v; }

    public String getMessage()                  { return message; }
    public void setMessage(String v)            { this.message = v; }

    public String getPath()                     { return path; }
    public void setPath(String v)               { this.path = v; }

    public LocalDateTime getTimestamp()         { return timestamp; }
    public void setTimestamp(LocalDateTime v)   { this.timestamp = v; }
}
