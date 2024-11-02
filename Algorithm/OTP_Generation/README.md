# One-time-passcode Generation

## Logic
- Private key
- `hash(private_key, time)`

## Implementation
- Agree on private key in 256-bit length
- Round epoch to nearest minute
    - `time.time() // 60`
- Cast to 32-bit integer
    - `int(epoch)`
- For each 32-bit in private key perform XOR with above integer and store as X
- Compute SHA256(X) and take the last 32-bit
- Cast the 32-bit into integer
- Rounding as needed
