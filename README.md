# OA_Oyster_ImageRecognition
Repository to count and measure oysters.
Keep .h5 data files and .jpg images out of the repository, or keep them in but make sure they get gitignored

To run in an environment that is already set up, nevigate to the directory where the repository lives on your local machine, and run a docker container from the image sccwrp/oyster-measurement:master (May be updated later. Don't know why I used master rather than latest...)

I like to run it in detached mode (-d) and leave it running, then open up a terminal in it later with docker exec -it (container name) /bin/bash)

docker container run -d -it --name (whatever you want to call it) -v /path/to/repo/:/home/ sccwrp/oyster-measurement:latest

Technically the mount doesn't have to be in the home directory of the container, but I would personally prefer to do it this way.

I figured the best way to have the code and the environment working together is to keep them separate, but run a docker container and have the repository set up as a volume when the container gets ran

Now, there are certain images that the code needs, which I have tried to keep out of the repo in order to keep it small so it doesn't take forever just to push and pull, so and images or data files will need to be kept in the directory where you are working on this thing but make sure that they get gitignored. 


I am thinking of sticking the necessary jpegs and model files in a folder inside the container in order to keep them completely out of the repository, and then we don't even have to worry about the .gitignore file




