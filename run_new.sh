angle=$1
extrinsic=$2
# angle, extrinsic, yaw, pitch, height

cp -r /tzofi/repos/scrape-bev-carla repo
cd /home/scrape-bev-carla/repo

mkdir "/tzofi/data/carla-zips/${2}${1}"
#mkdir "/tzofi/data/carla/${2}${1}/town03"
#mkdir "/tzofi/data/carla/${2}${1}/town05"
mkdir "/home/carla/town03"
mkdir "/home/carla/town05"

# TRAIN DATA
(/home/carla/CarlaUE4.sh 2>&1 > /dev/null &) && (sleep 15) && (python3 -u main.py scrape --outf="/home/carla/town03" --headless=True --rnd_seed=42 --filter_occluded=True --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" --map_name="Town03" > /results/scrape_town03.log &)

# TEST DATA
(/home/carla/CarlaUE4.sh --world-port=2040 2>&1 > /dev/null &) && (sleep 15) && (python3 -u main.py scrape --outf="/home/carla/town05" --headless=True --rnd_seed=42 --filter_occluded=True --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" --port=2040 --map_name="Town05" > /results/scrape_town05.log &)

sleep 64800s

tar -cvz -f /home/carla/town03.tar.gz /home/carla/town03
tar -cvz -f /home/carla/town05.tar.gz /home/carla/town05
mv /home/carla/town03.tar.gz /tzofi/data/carla-zips/${2}${1}/
mv /home/carla/town05.tar.gz /tzofi/data/carla-zips/${2}${1}/
