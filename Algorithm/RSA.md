# RSA

## Euclid's Algorithm

Given 2 Integers

```python
# Given a and b

while b != 0:
    r = a % b
    a, b = b, r
return a
```

## Extended Euclid's Algorithm

Formula

$GCD(a,b) = ax + by$

## Diophantine equation

Given $ f(x,y) $

Transforms $ f(x,y) $ to $ f'(k) $ where $ k \in \mathbb{Z} $

For example $ 3x + 4y = 5 $ becomes $ x = ak + b$ and $ y = ck + d $

## Trapdoor Function

## Assymetric Encryption in RSA

### Prime number (key) generation

1. Prime number generation
   - Let $ p = 61 $ and $ q = 53 $
2. Compute $n$
   - Let $ n = p \times q = 61 \times 53 = 3233$
3. Compute $ \phi(n)$
   - $ (p-1)(q-1) $
   - $ (61-1)(53-1) $
   - $ 60 \times 52 = 3120 $
4. Choose exponent $e$
   - where $ e $ is coprime with $ \phi(n) $
   - $ 1 < e < \phi(n) $
   - Let $e = 17$
5. Calculate private exponent $d$
   - where $e \times d \equiv 1  \bmod \phi(n)$
   - $ d = e^{-1} \bmod \phi(n) $
   - $ d = (17)^{-1} \bmod 3120$
   - $ d = 2753 $
6. Keys derived:
   - Public Key: $(e, n) = (17, 3233)$
   - Private Key: $(d, n) = (2753, 3233)$

### Encryption and Decryption

4 bytes can be represented as `uint32` and allows calculation

Or in 256-bit system, as `uint256`

Encryption

1. Assumes the data is $ M = 65$
2. Cryptogram $C$ is represented as $ M^e \bmod n$
    - $ 65^{17} \bmod 3233 $
    - $ C = 2790$

Decryption

1. Using $ M = C^d \bmod n$
2. $ M = 2790^{2753} \bmod 3233$
3. $ M = 65 $
