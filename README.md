# Roterman Viktor master's degree diploma👨‍🎓

Module for face recognition on video. Algoritm based on person trecking and detection face on tracked person.  
## Architecture

![ScreenShot](image/architecture.png)


# Befor using

install required libraries
```shell
pip install requirements.txt
```

in file diploma_docker_compose.yml change on your parametrs DB
```shell
POSTGRES_USER: viktor
POSTGRES_PASSWORD: 1452
POSTGRES_DB: diploma_db
```

init and up db
```shell
make istall
make up
```

and finally yoy can run main.py
```shell
python3 main.py
```

when scripts start, you must answer on 3 questions:

- 1 question
