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
   - SS = $ Pub\_a^{Priv\_b} \bmod p $ 
   - SS = $ 8^{15} \bmod 23 $
   - SS = $ 2 $

## Elliptic Curve Diffie Hellman
