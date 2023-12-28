# Description: Build docker image for attendance
# version is the latest
version=latest
docker build -t attendance .
docker tag attendance ahmedheakl/attendance:$version
docker push ahmedheakl/attendance:$version