package com.example.customerapi.health;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

// Custom health indicator to check if customer-status-service is accessible
// This will appear in the /actuator/health endpoint
@Component
public class CustomerStatusServiceHealthIndicator implements HealthIndicator {

    @Autowired
    private RestTemplate restTemplate;

    private static final String HEALTH_CHECK_URL = "http://customer-status-service/actuator/health";

    @Override
    public Health health() {
        try {
            // Try to call the health endpoint of customer-status-service
            // If successful, the service is UP
            String response = restTemplate.getForObject(HEALTH_CHECK_URL, String.class);
            
            // Service is reachable and responding
            return Health.up()
                    .withDetail("customer-status-service", "Available")
                    .withDetail("message", "Customer Status Service is healthy")
                    .build();
                    
        } catch (Exception e) {
            // Service is down or unreachable
            // This will show in actuator/health endpoint
            return Health.down()
                    .withDetail("customer-status-service", "Unavailable")
                    .withDetail("error", e.getMessage())
                    .withDetail("message", "Customer Status Service is not responding")
                    .build();
        }
    }
}
