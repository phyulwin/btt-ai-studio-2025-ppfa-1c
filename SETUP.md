# PPFA AI Studio 2025

Thank you for participating in AI Studio 2025 with us! We are looking forward to working together on this project.

## Getting started

### Docker

You'll be using Jupyter and Python for this project, but in order to make the process of getting set up a little simpler, we're suggesting that you use Docker. Docker is a containerization tool for bundling code and dependencies together. If you're familiar with package tools like uv, you're welcome to set this up in another way, but it will be easiest to follow this recommended process!

First, install [Docker Desktop](https://docs.docker.com/compose/install/). If you run into any problems with this step please ask us!

Once that's done, open up a terminal, change to the directory containing this README file, and run

```
    docker-compose up
```

The first time you do this, you'll see a lot of output text (it will be faster the subsequent times you run it). Eventually you will see some startup text followed by output that looks like:

```
    jupyter-1  | [I 2025-08-20 15:57:21.257 ServerApp]     http://127.0.0.1:8888/lab?token=211394ed0b20bf0e21ea1e85bbdf6ff9d1bc69ad18496b2e
    jupyter-1  | [I 2025-08-20 15:57:21.257 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

Open up the URL starting with http://127.0.0.1:8888/ in your browser. The "token" value above is just an example -- yours will be something different!

### Credentials

We will send you a file called `credential.json`. This is your permission for downloading the conversational data from our data store. Put it in the same directory as this file, but *do not commit it to git* and do not share it with anyone. (If you share it by mistake, just let us know and we'll invalidate it and create you a new one!)

### Adding libraries

We've tried to give you most of the tools you need for this project, but if you need additional libraries you can install them via Jupyter. However, you'll need to use a slightly different syntax than you may be used to. If you want to install scikit-learn, for instance, create a new cell and input:

```
    !cd config && uv add scikit-learn
```

(This is just an example as scikit-learn is already included.)
