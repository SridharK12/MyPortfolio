package com.example.customerapi.mapper;

import com.example.customerapi.dto.CustomerDetailsWithStatusResponse;
import com.example.customerapi.entity.Customer;

public class CustomerMapper {

    public static CustomerDetailsWithStatusResponse mapToDto(Customer customer, String status) {
        CustomerDetailsWithStatusResponse dto = new CustomerDetailsWithStatusResponse();
        dto.setCustomerId(customer.getCustomerId());
        dto.setCustomerName(customer.getCustomerName());
        dto.setCustomerDob(customer.getCustomerDob());

        dto.setCustomerSsn(customer.getCustomerSsn());     // JIRA-1
        dto.setCustomerState(customer.getCustomerState()); // JIRA-1

        dto.setModifiedBy(customer.getModifiedBy());
        dto.setModificationDate(customer.getModificationDate());
        dto.setStatus(status);
        return dto;
    }
}
