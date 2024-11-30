#

## Module

- Single Python file
- Function, class, variable, code

```python
# file: my_module.py
def greet(name):
    return f"Hello, {name}!"
```

- Regular import: `import my_module`
- Specific import: `from my_module import greet`

## Package

- Collection of Python modules organized in a directory
- Executes `__init__.py`

```
my_package/
├── __init__.py
├── module1.py
└── module2.py
```

```python
# __init__.py
from . import module1
from . import module2
```

- Import a package: `import my_package`
- Import module from package: `from my_package import module1`
- Import specific item from module in package: `from my_package.module1 import function_name`

## Rules

### Absolute Import

```python
from my_package.module1 import some_function
```

### Relative Import

```python
from .module1 import some_function
from ..module2 import another_function
```

## Execution

### Module

```python
# file: my_module.py
def greet(name):
    return f"Hello, {name}!"

if __name__ == "__main__":
    print(greet("Alice"))
```

```bash
python3 my_module.py
```

### Package

Looks for `if __name__ == "__main__":`

```bash
python3 -m my_module
```

```bash
python3 -m my_package.module1
```
