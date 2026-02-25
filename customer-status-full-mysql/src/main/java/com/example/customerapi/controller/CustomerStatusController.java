package com.example.customerapi.controller;

import com.example.customerapi.dto.CustomerStatusResponse;
import com.example.customerapi.service.CustomerStatusService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/api/customers")
public class CustomerStatusController {

    @Autowired
    private CustomerStatusService statusService;

    /**
     * GET /api/customers/{id}
     * Returns CustomerStatusResponse { customerId, status }
     */
    @GetMapping("/{id}")
    public ResponseEntity<CustomerStatusResponse> getStatus(@PathVariable("id") Long id) {
        try {
            CustomerStatusResponse resp = statusService.getStatusByCustomerId(id);
            return ResponseEntity.ok(resp);
        } catch (RuntimeException e) {
            // status not found
            return ResponseEntity.notFound().build();
        } catch (Exception e) {
            // generic error — return 500
            return ResponseEntity.status(500).build();
        }
    }
}
