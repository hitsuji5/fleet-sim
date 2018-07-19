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

### 6. Create Mesh Map
```commandline
python movi/preprocessing/create_tt_map.py ./data
```


## Run Simulation
### 1. Run OSRM container
```commandline
cd osrm
docker run -t -i -p 5000:5000 -v $(pwd):/data osrm/osrm-backend osrm-routed --algorithm mld /data/NewYork.osrm
```

### 2. Training
```commandline
sh bin/train.sh
```


### 3. Evaluation
```commandline
sh bin/eval.sh
```