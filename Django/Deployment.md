# Deployment

## WSGI

`wsgi.py`

- Gunicorn
  - `gunicorn your_project_name.wsgi:application --bind 0.0.0.0:8000 --workers 9 --timeout 30`
- uWSGI
- Apache mod_wsgi

## ASGI

`asgi.py`

- Uvicorn
  - `uvicorn myproject.asgi:application --host 0.0.0.0 --port 8000`
- Daphne
- Hypercorn
