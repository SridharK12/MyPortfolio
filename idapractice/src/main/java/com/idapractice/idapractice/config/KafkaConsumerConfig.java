package com.idapractice.idapractice.config;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.kafka.listener.DeadLetterPublishingRecoverer;
import org.springframework.kafka.listener.DefaultErrorHandler;
import org.springframework.util.backoff.FixedBackOff;

import java.util.Map;

/**
 * Kafka consumer error handling.
 *
 * Flow when a consumer listener throws:
 *   1. Spring Kafka retries up to MAX_RETRIES times with BACKOFF_MS delay between attempts.
 *   2. If still failing after all retries, DeadLetterPublishingRecoverer sends the
 *      record to <original-topic>.DLT on the same partition.
 *   3. The DLT can be monitored/alerted on and re-processed manually.
 *
 * Non-retryable exceptions (IllegalArgumentException, ClassCastException) skip
 * retries and go straight to the DLT — retrying bad data is pointless.
 */
@Configuration
public class KafkaConsumerConfig {

    private static final Logger log = LoggerFactory.getLogger(KafkaConsumerConfig.class);

    private static final long BACKOFF_MS  = 1_000L; // 1 second between retries
    private static final long MAX_RETRIES = 3L;

    @Bean
    public DefaultErrorHandler kafkaErrorHandler(KafkaTemplate<String, Object> kafkaTemplate) {

        // Routes failed records to <topic>.DLT on the same partition number
        DeadLetterPublishingRecoverer recoverer = new DeadLetterPublishingRecoverer(
                Map.of(Object.class, kafkaTemplate),
                (record, ex) -> {
                    log.error("Sending record to DLT — topic={} partition={} offset={} key={}",
                            record.topic(), record.partition(), record.offset(), record.key(), ex);
                    return new org.apache.kafka.common.TopicPartition(
                            record.topic() + ".DLT", record.partition());
                });

        DefaultErrorHandler handler = new DefaultErrorHandler(
                recoverer,
                new FixedBackOff(BACKOFF_MS, MAX_RETRIES));

        // These exceptions indicate bad data — retrying won't help, go straight to DLT
        handler.addNotRetryableExceptions(
                IllegalArgumentException.class,
                ClassCastException.class,
                NullPointerException.class);

        return handler;
    }
}
