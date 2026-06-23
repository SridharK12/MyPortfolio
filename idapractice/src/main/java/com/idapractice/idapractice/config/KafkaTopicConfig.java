package com.idapractice.idapractice.config;

import org.apache.kafka.clients.admin.NewTopic;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.config.TopicBuilder;

/**
 * Declares all payment lifecycle topics (and their Dead Letter Topics).
 * KafkaAdmin creates them on startup if they do not already exist.
 *
 * Main topics       → published by PaymentOutboxPublisher
 * *.DLT topics      → messages land here after 3 failed consumer retries
 *                     (configured in KafkaConsumerConfig)
 *
 * Partition count: 3 (allows 3 parallel consumer instances per topic).
 * Replica count  : 1 for local/dev; override to 3 on AWS MSK via env var.
 */
@Configuration
public class KafkaTopicConfig {

    private static final int PARTITIONS = 3;
    private static final int REPLICAS   = 1; // Set to 3 in prod via MSK config

    @Bean public NewTopic paymentCreated()          { return topic("payment-created"); }
    @Bean public NewTopic paymentUpdated()          { return topic("payment-updated"); }
    @Bean public NewTopic paymentApproved()         { return topic("payment-approved"); }
    @Bean public NewTopic paymentRejected()         { return topic("payment-rejected"); }
    @Bean public NewTopic paymentCancelled()        { return topic("payment-cancelled"); }

    // Dead Letter Topics — one per main topic
    @Bean public NewTopic paymentCreatedDlt()       { return topic("payment-created.DLT"); }
    @Bean public NewTopic paymentUpdatedDlt()       { return topic("payment-updated.DLT"); }
    @Bean public NewTopic paymentApprovedDlt()      { return topic("payment-approved.DLT"); }
    @Bean public NewTopic paymentRejectedDlt()      { return topic("payment-rejected.DLT"); }
    @Bean public NewTopic paymentCancelledDlt()     { return topic("payment-cancelled.DLT"); }

    private NewTopic topic(String name) {
        return TopicBuilder.name(name)
                .partitions(PARTITIONS)
                .replicas(REPLICAS)
                .build();
    }
}
