package com.example.customerapi.health;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestTemplate;

@Component
public class CustomerStatusServiceHealthIndicator implements HealthIndicator {

    @Autowired
    private RestTemplate restTemplate;

    private static final String HEALTH_CHECK_URL =
            "http://customer-status-service/actuator/health";

    @Override
    public Health health() {
        try {
            restTemplate.getForObject(HEALTH_CHECK_URL, String.class);
            return Health.up()
                    .withDetail("customer-status-service", "UP")
                    .build();
        } catch (Exception e) {
            return Health.down()
                    .withDetail("customer-status-service", "DOWN")
                    .withDetail("error", e.getMessage())
                    .build();
        }
    }
}