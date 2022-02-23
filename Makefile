docker-build: 
	docker image build -t pymc3_sunode .

docker-run: 
	docker run -p 8888:8888 -v $(pwd):/home/jovyan/work pymc3_sunode

docker-restart-container:
	docker rmi pymc3_sunode
	docker-build

docker-stop-container:
	docker stop $(docker ps -a -q --filter ancestor=pymc3_sunode)