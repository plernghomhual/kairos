## Lesson: Tuple Return Signature Changes

### Anti-Pattern
Adding return values to a widely-used function (`fetch_live_data`) broke downstream tests because Python raises `ValueError: too many values to unpack` when the caller expects fewer values.

### Pattern
When adding arguments/return values to public functions, update ALL callers and tests in the same commit. For async coroutines that are mocked in tests, ensure the mock response schema matches what the modified code expects.

### Trigger
Modifying return signature of a function called in multiple files.
