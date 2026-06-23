package com.idapractice.idapractice.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.annotation.EnableScheduling;

/**
 * Enables Spring's @Scheduled support, required by PaymentOutboxPublisher.
 *
 * Kept in its own class so it can be excluded from test slices
 * (e.g. @WebMvcTest) without touching the main application class.
 */
@Configuration
@EnableScheduling
public class SchedulingConfig {
}
