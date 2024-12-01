# Routing in Flask

## Blueprint

```
my_flask_app/
|-- app.py
|-- blueprints/
|   |-- __init__.py
|   |-- auth.py
|   |-- admin.py
|-- templates/
|-- static/
```

```python
from flask import Flask
from blueprints.auth import auth_bp
from blueprints.admin import admin_bp

app = Flask(__name__)

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/auth')
app.register_blueprint(admin_bp, url_prefix='/admin')

if __name__ == '__main__':
    app.run(debug=True)
```

```python
from flask import Blueprint, jsonify

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    return jsonify({"message": "Login endpoint"})

@auth_bp.route('/register', methods=['POST'])
def register():
    return jsonify({"message": "Register endpoint"})
```

```python
from flask import Blueprint, jsonify

admin_bp = Blueprint('admin', __name__)

@admin_bp.route('/dashboard', methods=['GET'])
def dashboard():
    return jsonify({"message": "Admin Dashboard"})
```

## Blueprint With Versioning

```
my_flask_app/
|-- app.py
|-- api/
|   |-- v1/
|   |   |-- __init__.py
|   |   |-- users.py
|   |-- v2/
|       |-- __init__.py
|       |-- users.py
```

```python
from flask import Flask
from api.v1 import v1_bp
from api.v2 import v2_bp

app = Flask(__name__)

# Register versioned Blueprints
app.register_blueprint(v1_bp, url_prefix='/api/v1')
app.register_blueprint(v2_bp, url_prefix='/api/v2')

if __name__ == '__main__':
    app.run(debug=True)
```

```python
from flask import Blueprint

v1_bp = Blueprint('v1', __name__)

from .users import users_bp

v1_bp.register_blueprint(users_bp, url_prefix='/users')
```

```python
from flask import Blueprint, jsonify

users_bp = Blueprint('users_v1', __name__)

@users_bp.route('/')
def get_users():
    return jsonify({"message": "List of users (v1)"})
```
