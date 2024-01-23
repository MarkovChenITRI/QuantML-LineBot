docker rm -f quantml_container
docker build -f ./docker/Dockerfile . -t quantml
docker container run --name quantml_container -p 7860:7860 quantml
