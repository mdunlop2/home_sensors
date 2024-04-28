# Home Sensors

Build a simple model that can predict whether a household is single or multiple occupancy using data from motion sensors installed in the home.

## Usage

```
git clone git@github.com:mdunlop2/home_sensors.git
cd home_sensors
docker run -p 8888:8888 -v $PWD:/home/jovyan/work docker.io/mdunlop/home_sensors:latest
```