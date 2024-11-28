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
