package com.idapractice.idapractice.component;

import com.idapractice.idapractice.dto.PaymentResponseDTO;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.kafka.support.KafkaHeaders;
import org.springframework.messaging.handler.annotation.Header;
import org.springframework.messaging.handler.annotation.Payload;
import org.springframework.stereotype.Component;

/**
 * Kafka consumers for all five payment lifecycle topics.
 *
 * Each listener receives a fully-typed PaymentResponseDTO.
 * JSON → object conversion is handled by JsonDeserializer
 * (configured in application.properties).
 *
 * Error handling (retries + DLT routing) is wired globally in KafkaConsumerConfig.
 * Individual listeners do NOT catch exceptions — they let them propagate so the
 * error handler can apply the retry / DLT policy.
 *
 * In a microservices architecture these listeners would live in separate services.
 * They are co-located here for simplicity and can be extracted without service changes.
 */
@Component
public class PaymentEventConsumer {

    private static final Logger log = LoggerFactory.getLogger(PaymentEventConsumer.class);

    @KafkaListener(topics = "payment-created", groupId = "payment-group")
    public void onPaymentCreated(
            @Payload PaymentResponseDTO payment,
            @Header(KafkaHeaders.RECEIVED_KEY) String key,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {

        log.info("[{}] CREATED — id={} from={} to={} amount={}",
                topic, key, payment.getFromAccount(),
                payment.getToAccount(), payment.getAmount());

        // TODO: trigger fraud screening, send maker confirmation email
    }

    @KafkaListener(topics = "payment-updated", groupId = "payment-group")
    public void onPaymentUpdated(
            @Payload PaymentResponseDTO payment,
            @Header(KafkaHeaders.RECEIVED_KEY) String key,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {

        log.info("[{}] UPDATED — id={} amount={} updatedAt={}",
                topic, key, payment.getAmount(), payment.getUpdatedAt());

        // TODO: invalidate cache, re-run limit checks
    }

    @KafkaListener(topics = "payment-approved", groupId = "payment-group")
    public void onPaymentApproved(
            @Payload PaymentResponseDTO payment,
            @Header(KafkaHeaders.RECEIVED_KEY) String key,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {

        log.info("[{}] APPROVED — id={} from={} to={} amount={}",
                topic, key, payment.getFromAccount(),
                payment.getToAccount(), payment.getAmount());

        // TODO: call core-banking transfer API, send approval email
    }

    @KafkaListener(topics = "payment-rejected", groupId = "payment-group")
    public void onPaymentRejected(
            @Payload PaymentResponseDTO payment,
            @Header(KafkaHeaders.RECEIVED_KEY) String key,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {

        log.info("[{}] REJECTED — id={} remarks={}",
                topic, key, payment.getRemarks());

        // TODO: notify maker with rejection reason, release reserved funds
    }

    @KafkaListener(topics = "payment-cancelled", groupId = "payment-group")
    public void onPaymentCancelled(
            @Payload PaymentResponseDTO payment,
            @Header(KafkaHeaders.RECEIVED_KEY) String key,
            @Header(KafkaHeaders.RECEIVED_TOPIC) String topic) {

        log.info("[{}] CANCELLED — id={} from={} amount={}",
                topic, key, payment.getFromAccount(), payment.getAmount());

        // TODO: release reserved funds, remove from approver queue
    }
}
