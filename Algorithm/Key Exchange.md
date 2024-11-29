# Key Exchange

## Diffie Hellman

1. Choose Public Parameters
   - Let Prime $ p = 23 $
   - Let Primitive Root Modulo $ p,g = 5 $
2. Private Keys
   - Let a = 6
   - let b = 15
3. Compute Public Keys
   - Public Key = $ g^{Priv Key} \bmod p $
   - A = $ g^a \bmod p = 5^{6} \bmod 23 = 8 $
   - B = $ g^b \bmod p = 5^{6} \bmod 23 = 19 $
4. Exchange Public Keys
5. Compute Shared Secret
   - SS = $ Pub_a^{Priv_b} \bmod p $
   - SS = $ 8^{15} \bmod 23 $
   - SS = $ 2 $

## Elliptic Curve Diffie Hellman

Three rules

- Point addition: $ P + Q$
  - $ P \neq Q $
    - $ x_3 = m^2 - x_1 - x_2 \bmod p$
    - $ y_3 = m \times (x_1 - x_3) - y_1 \bmod p$
  - $ P = Q $
    - Point doubling
  - $ x_1 = x_2 $ and $ y_1 = -y_2 \bmod p $
    - $ P + Q = 0 $
- Point doubling: $ P + P $
  - $ x_3 = m^2 - 2x_1 \bmod p$
  - $ y_3 = m \times (x_1 - x_3) - y_1 \bmod p $
- Point multipication: $ k \times p $
    - TBD
- Modular Inverse
    - $ n \times x \equiv 1 \bmod p$

Elliptic Curve Equation: $ y^2 = x^3 + ax + b \bmod p $

- Coefficients: $ a,b $
- Large prime: $ p $
