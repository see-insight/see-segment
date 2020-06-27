# Azure Set Up
The following document provides instructions for running see-segment in containers on the azure cloud platform. 
If you get stuck try consulting https://zero-to-jupyterhub.readthedocs.io/en/latest/microsoft/step-zero-azure.html

## Step 1
Create an Azure account:

https://azure.microsoft.com/en-us/free/

## Step 2
Install the azure cli:

https://docs.microsoft.com/en-us/cli/azure/install-azure-cli?view=azure-cli-latest

## Step 3
Login to the azure cli:

`az login`
Then login in the browser window that opens.

## Step 4
Create a Kubernetes Cluster in Azure:

https://portal.azure.com/#create/microsoft.aks

Use default settings. The only exceptions are to lower the node count to 2 nodes on the first screen.
If you have not already requested an increase in available cpu resources on azure. The second exception is
in the network tab to enable HTTP application routing. This may not be necessary, but all testing was done with it selected.

Click review and create. You will see an error prompting you to enter missing data. You can fill in the resource group, clustername, and dns prefix to be anything you want.

Then click review and create again to create your cluster. The cluster will take a few minutes to get up and running. 

## Step 5
Get credentials to run kubectl:

`az aks get-credentials --name <CLUSTER-NAME> --resource-group <RESOURCE-GROUP-NAME> --output table`
Note: You must replace <CLUSTER-NAME> and <RESOURCE-GROUP-NAME> with the values you used in the previous step.

## Step 6
Verify your cluster is working:

`kubectl get node`
When the cluster is ready you should see some nodes with a status of ready.

## Step 7
Once kubernetes and cluster are set up and working to start the services run

`kubectl apply -f server_service.yaml
kubectl apply -f server.yaml
kubectl apply -f segmentation_job.yaml`

To see running service and the external ip address of the server run:

`kubectl get services`

To see running pods run:

`kubectl get pods`

## Step 8
Eventually the service will be given an external ip address, keep running kubectl get services until you see one.
The web server will now be accessible at this address. Note at startup it might have a very slow load time.
