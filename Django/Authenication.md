# Authenication System

##

## User Object

- `User` model provides basic information about users

## Login Using Template

## User Login View

```python
request.POST['username']
request.POST['password']
user = authenticate(username=username, password=password)
login(request, user)
```

## Check Session

```python
sessionid = request.COOKIES.get('sessionid')
session = request.session.items()
```

## Permission

Permission on objects (model)

- View
- Add
- Change
- Delete
