# To upload a docker image to dockerhub
sudo docker login -u seesegment --password-stdin REQUEST_THE_PASSWORD

## Updating the Docker images
When updating docker images make sure to modify the yaml files to use the latest verion of the container

After making and testing any code changes locally, when want to update the containers on dockerhub

### run inside the seesegment directory
sudo docker build -t seesegment/seesegment:0.0.4 .
sudo docker push seesegment/seesegment:0.0.4

### run inside the seesegment/see_server directory
sudo docker build -t seesegment/seeserver:0.0.4 .
sudo docker push seesegment/seeserver:0.0.4