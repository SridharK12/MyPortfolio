package com.idapractice.idapractice.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

public class AuthorizationRequestDTO {

    @NotBlank(message = "status is mandatory for authorization")
    @Pattern(
        regexp = "(?i)APPROVED|REJECTED",
        message = "status must be APPROVED or REJECTED"
    )
    private String status;

    @Size(max = 500, message = "remarks must not exceed 500 characters")
    private String remarks;

    public String getStatus()           { return status; }
    public void setStatus(String v)     { this.status = v; }

    public String getRemarks()          { return remarks; }
    public void setRemarks(String v)    { this.remarks = v; }
}
