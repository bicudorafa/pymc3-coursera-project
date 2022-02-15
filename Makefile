setup: 
	docker build -t pymc3_sunode_env .

run: 
	docker run -p 8888:8888 -v $(pwd):/home/jovyan/work bicudorafa/jupyter_pymc3_env