package com.idapractice.idapractice.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/**
 * Declares the payment-approved / payment-rejected topics so KafkaAdmin
 * creates them automatically on startup. 1 partition / 1 replica is fine
 * for a single-broker local dev setup.
 */
@Configuration
public class KafkaTopicConfig {

    @Bean
    public NewTopic paymentApprovedTopic() {
        return TopicBuilder.name("payment-approved")
                .partitions(1)
                .replicas(1)
                .build();
    }

    @Bean
    public NewTopic paymentRejectedTopic() {
        return TopicBuilder.name("payment-rejected")
                .partitions(1)
                .replicas(1)
                .build();
    }
}
