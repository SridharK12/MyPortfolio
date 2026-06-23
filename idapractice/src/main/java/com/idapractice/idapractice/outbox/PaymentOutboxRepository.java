package com.idapractice.idapractice.outbox;

import com.idapractice.idapractice.entity.PaymentOutboxEvent;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface PaymentOutboxRepository extends JpaRepository<PaymentOutboxEvent, Long> {

    /**
     * Fetches unpublished events ordered by creation time (oldest first),
     * with a page-size cap to avoid loading the entire backlog at once.
     *
     * The Pageable param lets the publisher limit to, e.g. 50 events per run.
     */
    List<PaymentOutboxEvent> findByPublishedFalseOrderByCreatedAtAsc(Pageable pageable);
}
