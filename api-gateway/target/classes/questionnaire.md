Here’s a crisp, interview-ready Q & A based exactly on your code.
Memorize these — they map 1:1 to what interviewers usually ask 👇
________________________________________
Q1. What is this filter used for?
A:
It is a Spring Cloud Gateway GlobalFilter used to generate or propagate a Correlation ID, add it to outgoing requests, and enable end-to-end request tracing and logging across microservices.
________________________________________
Q2. Why did you use GlobalFilter instead of a route filter?
A:
Because correlation ID and logging are cross-cutting concerns and must apply to all routes consistently, not route-by-route.
________________________________________
Q3. Why does the filter implement Ordered?
A:
To control execution order.
Using Ordered.HIGHEST_PRECEDENCE ensures the correlation ID is created before any other filter executes, so all downstream filters and services can use it.
________________________________________
Q4. How is the Correlation ID generated?
A:
The filter first checks for X-Correlation-Id in request headers.
If it’s missing or blank, a UUID is generated and used as the correlation ID.
________________________________________
Q5. Why do you mutate the request?
A:
In WebFlux, request objects are immutable.
To add a header, we must create a mutated copy of the request using request.mutate() and attach it back to the exchange.
________________________________________
Q6. Why is ServerWebExchange used here?
A:
ServerWebExchange represents the entire reactive HTTP interaction (request + response) and allows mutation and propagation of the modified request downstream.
________________________________________
Q7. Why can’t MDC alone be used in reactive applications?
A:
MDC relies on ThreadLocal, but reactive pipelines can switch threads.
So MDC values may be lost unless explicitly managed.
________________________________________
Q8. How did you solve MDC issues in a reactive flow?
A:
•	Stored correlationId in Reactor Context using contextWrite()
•	Copied it into MDC during signal processing using doOnEach()
•	Cleared MDC in doFinally() to avoid leaks
________________________________________
Q9. Why is contextWrite() important here?
A:
It allows propagating request-scoped data (correlationId) across asynchronous reactive operators, independent of thread execution.
________________________________________
Q10. Why is doOnEach() used instead of doOnNext()?
A:
Because it captures all signal types (onNext, onComplete, onError) and ensures MDC is populated consistently whenever logging occurs.
________________________________________
Q11. Why is doFinally() used?
A:
doFinally() executes exactly once, regardless of success, failure, or cancellation, making it ideal for:
•	Logging total request duration
•	Cleaning up MDC data
________________________________________
Q12. How is request duration calculated?
A:
By capturing startTime before the chain executes and calculating the difference when the request completes.
________________________________________
Q13. Why do you default the response status to 500?
A:
In some failure scenarios, the response status may be null.
Using 500 ensures safe and consistent logging.
________________________________________
Q14. How does this help in production monitoring?
A:
It enables:
•	Distributed tracing
•	Centralized log correlation (ELK, Splunk, Loki)
•	Latency analysis
•	Debugging across microservices
________________________________________
Q15. Can downstream services access the correlation ID?
A:
Yes. The correlation ID is propagated via the HTTP header, so downstream services can log and forward it further.
________________________________________
Q16. How would you improve this in a real system?
A:
•	Use OpenTelemetry / Spring Cloud Sleuth
•	Integrate with Micrometer Tracing
•	Avoid manual MDC handling where possible
________________________________________
🔑 Final one-line answer to remember
“This GlobalFilter provides reactive-safe correlation ID propagation and request lifecycle logging at the gateway, ensuring end-to-end traceability across microservices.”
If you want, I can also:
•	🎯 Compress this into 5 must-answer questions
•	🎯 Turn it into a whiteboard explanation
•	🎯 Add tricky follow-up questions interviewers ask
Just tell me.

