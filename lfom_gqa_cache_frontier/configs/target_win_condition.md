# Target win condition

The large-model result is strong only if LFOM-GQA or LFOM-MQA:

1. matches or beats MHA on validation CE and long-context CE,
2. uses 25%-50% of the K/V cache,
3. beats the nonrecurrent MLP repair control on paired seeds,
4. improves the quality-per-cache or quality-per-throughput frontier.

If the condition fails, report the current result as a small-scale mechanism and do not claim a top-tier cache-compression result.
