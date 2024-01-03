# Description: Build docker image for attendance
version=latest
docker build -t ahmedheakl/attendance:$version .
docker push ahmedheakl/attendance:$version