install:
	sudo apt install docker-compose && \
	sudo usermod -aG docker $$USER && \
	sudo service docker restart

rm:
	docker compose stop && \
	docker compose rm && \
	sudo rm -rf /home/viktor/projects/diploma_fr/data/pgdata

up:
	docker compose -f diploma_docker_compose.yml up --force-recreate