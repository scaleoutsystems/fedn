find . -path "*/migrations/*.py" -not -name "__init__.py" -delete
find . -path "*/migrations/*.pyc"  -delete
sleep 5
python3 -m pip install -r requirements.txt
python3 manage.py makemigrations combiner
python3 manage.py migrate
python3 manage.py loaddata seed.json
python3 manage.py runserver ${CONTROLLER_HOST}:${CONTROLLER_PORT}