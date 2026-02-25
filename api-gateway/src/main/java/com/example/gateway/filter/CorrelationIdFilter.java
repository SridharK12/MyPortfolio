package com.example.gateway.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.core.Ordered;
import org.springframework.http.server.reactive.ServerHttpRequest;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.util.UUID;
// MDC is mapped diagnostic content

@Component
public class CorrelationIdFilter implements GlobalFilter, Ordered {

    private static final Logger log = LoggerFactory.getLogger(CorrelationIdFilter.class);

    public static final String CORRELATION_ID_HEADER = "X-Correlation-Id";
    private static final String MDC_KEY = "correlationId";

    @Override
    public Mono<Void> filter(ServerWebExchange exchange,
                             org.springframework.cloud.gateway.filter.GatewayFilterChain chain) {

        long startTime = System.currentTimeMillis();

        ServerHttpRequest request = exchange.getRequest();

        String correlationId_1 = request.getHeaders().getFirst(CORRELATION_ID_HEADER);
        if (correlationId_1 == null || correlationId_1.isBlank()) {
            correlationId_1 = UUID.randomUUID().toString();
        }
        final String correlationId= correlationId_1;

        ServerHttpRequest mutatedRequest = request.mutate()
                .header(CORRELATION_ID_HEADER, correlationId)
                .build();

        return chain.filter(exchange.mutate().request(mutatedRequest).build())
                .contextWrite(ctx -> ctx.put(MDC_KEY, correlationId))
                .doOnEach(signal -> {
                    if (signal.isOnNext() || signal.isOnComplete()) {
                        MDC.put(MDC_KEY, correlationId);
                    }
                })
                .doFinally(signalType -> {
                    long durationMs = System.currentTimeMillis() - startTime;

                    int status = exchange.getResponse().getStatusCode() != null
                            ? exchange.getResponse().getStatusCode().value()
                            : 500;

                    log.info(
                            "Gateway completed | method={} path={} status={} durationMs={}",
                            request.getMethod(),
                            request.getURI().getPath(),
                            status,
                            durationMs
                    );

                    MDC.clear();
                });
    }

    @Override
    public int getOrder() {
        return Ordered.HIGHEST_PRECEDENCE;
    }
}