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

## Service

```bash
[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=your_user
Group=www-data
WorkingDirectory=/path/to/your/django_project
ExecStart=/path/to/venv/bin/gunicorn --workers 9 --bind 127.0.0.1:8000 your_project_name.wsgi:application

[Install]
WantedBy=multi-user.target
```

## Nginx Reverse Proxy

```bash
server {
    listen 80;
    server_name yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```