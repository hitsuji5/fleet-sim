# MOVI: Fleet Management Simulation Framework

## Setup

### 1. Install Modules
```commandline
pip install -r requirements.txt
```

### 2. Download OSM Data
```commandline
wget https://download.bbbike.org/osm/bbbike/NewYork/NewYork.osm.pb -P osrm
```

### 3. Preprocess OSM Data
```commandline
cd osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-extract -p /opt/car.lua /data/NewYork.osm.pbf
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-partition /data/NewYork.osrm
docker run -t -v $(pwd):/data osrm/osrm-backend osrm-customize /data/NewYork.osrm
```

### 4. Download Trip Data
```commandline
wget https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2016-05.csv -P trip_records
wget https://s3.amazonaws.com/nyc-tlc/trip+data/green_tripdata_2016-05.csv -P trip_records
```

### 5. Preprocess Trip Data
```commandline
python movi/preprocessing/preprocess_nyc_dataset.py trip_records/ --month 2016-05
python movi/preprocessing/create_backlog.py trip_records/trips_2016-05.csv
python movi/preprocessing/create_prediction.py
```

### 6. Create Trip Time Map
```commandline
python movi/preprocessing/create_tt_map.py ./data
```


## Quick Start
### 1. Build Simulator Image
```commandline
docker-compose build sim
```

### 2. Run Simulation
```commandline
docker-compose up -d
```