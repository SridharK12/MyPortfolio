package com.idapractice.idapractice.outbox;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.idapractice.idapractice.dto.PaymentResponseDTO;
import com.idapractice.idapractice.entity.PaymentOutboxEvent;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.data.domain.PageRequest;
import org.springframework.kafka.core.KafkaTemplate;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;
import org.springframework.transaction.annotation.Transactional;

import java.time.LocalDateTime;
import java.util.List;

/**
 * Polls the payment_outbox table and publishes unpublished events to Kafka.
 *
 * Runs every 5 seconds (configurable via app.outbox.poll-delay-ms).
 * Processes at most 50 events per run to cap latency and memory usage.
 *
 * Retry behaviour:
 *  If kafkaTemplate.send() fails (Kafka unavailable), the row stays
 *  published=false and is retried on the next scheduled run.
 *  This means at-least-once delivery — consumers must be idempotent.
 */
@Component
public class PaymentOutboxPublisher {

    private static final Logger log = LoggerFactory.getLogger(PaymentOutboxPublisher.class);
    private static final int BATCH_SIZE = 50;

    private final PaymentOutboxRepository outboxRepository;
    private final KafkaTemplate<String, Object> kafkaTemplate;
    private final ObjectMapper objectMapper;

    public PaymentOutboxPublisher(PaymentOutboxRepository outboxRepository,
                                  KafkaTemplate<String, Object> kafkaTemplate,
                                  ObjectMapper objectMapper) {
        this.outboxRepository = outboxRepository;
        this.kafkaTemplate    = kafkaTemplate;
        this.objectMapper     = objectMapper;
    }

    @Scheduled(fixedDelayString = "${app.outbox.poll-delay-ms:5000}")
    @Transactional
    public void publishPendingEvents() {

        List<PaymentOutboxEvent> pending = outboxRepository
                .findByPublishedFalseOrderByCreatedAtAsc(PageRequest.of(0, BATCH_SIZE));

        if (pending.isEmpty()) {
            return;
        }

        log.debug("Outbox poll: {} event(s) to publish", pending.size());

        for (PaymentOutboxEvent event : pending) {
            try {
                PaymentResponseDTO payload =
                        objectMapper.readValue(event.getPayload(), PaymentResponseDTO.class);

                // .get() blocks until the broker acknowledges — gives us a clear
                // success/failure signal within this scheduled invocation.
                kafkaTemplate.send(event.getTopic(), event.getMessageKey(), payload).get();

                event.setPublished(true);
                event.setPublishedAt(LocalDateTime.now());
                outboxRepository.save(event);

                log.info("Published outbox event id={} topic={} paymentId={}",
                        event.getId(), event.getTopic(), event.getPaymentId());

            } catch (Exception ex) {
                // Log and continue — leave row unpublished for next poll cycle.
                // Do NOT rethrow: one failed event must not block the others.
                log.error("Failed to publish outbox event id={} paymentId={} topic={} — will retry",
                        event.getId(), event.getPaymentId(), event.getTopic(), ex);
            }
        }
    }
}
