package com.example.customerapi.service;
import com.example.customerapi.dto.CustomerStatusResponse;
import com.example.customerapi.entity.CustomerStatus;
import com.example.customerapi.repository.CustomerStatusRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class CustomerStatusService {

    @Autowired
    private CustomerStatusRepository repository;
    

    private static final Logger log = LoggerFactory.getLogger(CustomerStatusService.class);

    /**
     * Returns the CustomerStatusResponse for the given customerId.
     * Throws RuntimeException if not found (controller converts to 404).
     */
    public CustomerStatusResponse getStatusByCustomerId(Long customerId) {
    	
    	log.info("ENTERED CustomerStatusService.getStatusByCustomerId()");
        Optional<CustomerStatus> opt = repository.findById(customerId);

        if (!opt.isPresent()) {
            throw new RuntimeException("Status not found for customerId: " + customerId);
        }
        
        log.info("In customer status service fetching custome rstatus {}", customerId);
        CustomerStatus cs = opt.get();
        CustomerStatusResponse resp = new CustomerStatusResponse();
        resp.setCustomerId(cs.getCustomerId());
        resp.setStatus(cs.getStatus());
        return resp;
    }

    /**
     * Save or update status for a customer (useful later).
     */
    public CustomerStatus saveOrUpdateStatus(CustomerStatus customerStatus) {
        return repository.save(customerStatus);
    }
}
