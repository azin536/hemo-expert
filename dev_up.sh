docker build --tag "exp:devel" --file Dockerfile.dev .

echo "running the container ..."
CODE=$(realpath .)
#docker run --name exp-devel -d -v "$CODE:/code" -p 8888:8888 "exp:devel"
docker run --name exp-devel -it -v "$CODE:/code" -p 8888:8888 exp:devel bash
# and then run this command `jupyter notebook --allow-root --no-browser --ip 0.0.0.0 --port 8888`