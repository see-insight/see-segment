# For see-segment maintainers only:
## To upload a Docker image to Docker Hub:
sudo docker login -u seesegment --password-stdin REQUEST_THE_PASSWORD

## Updating the Docker images:
When updating docker images make sure to modify the yaml files to use the latest verion of the container

After making and testing any code changes locally, when want to update the containers on dockerhub

### Build and push updated see-segment image:
Run inside the see-segment directory

`sudo docker build -t seesegment/seesegment:0.0.4`

`sudo docker push seesegment/seesegment:0.0.4`

### Build and push updated server image:
Run inside the seesegment/see_server directory

`sudo docker build -t seesegment/seeserver:0.0.4`

`sudo docker push seesegment/seeserver:0.0.4`
