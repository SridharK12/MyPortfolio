package com.example.gateway.filter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.cloud.gateway.filter.GlobalFilter;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import reactor.core.publisher.Mono;

@Configuration
public class LoggingFilter {

    private static final Logger log = LoggerFactory.getLogger(LoggingFilter.class);

    @Bean
    public GlobalFilter logFilter() {
        return (exchange, chain) -> {
            log.info("Request: {} {}", exchange.getRequest().getMethod(), exchange.getRequest().getURI());
            return chain.filter(exchange)
                    .then(Mono.fromRunnable(() ->
                            log.info("Response Status: {}", exchange.getResponse().getStatusCode())
                    ));
        };
    }
}
