angle=$1
extrinsic=$2
# angle, extrinsic, yaw, pitch, height

#cp -r /tzofi/repos/scrape-bev-carla repo
cd /home/abhinav/project/CARLA_rendering

mkdir "/media/abhinav/baap2/abhinav/datasets/viewpoint${2}${1}"
mkdir "/home/abhinav/project/CARLA_rendering/town03"
mkdir "/home/abhinav/project/CARLA_rendering/town05"

# TRAIN DATA
(/home/abhinav/project/CARLA_0.9.14/CarlaUE4.sh 2>&1 > /dev/null &) && (sleep 15) && (python3 -u main.py scrape --outf="/home/abhinav/project/CARLA_rendering/town03" --headless=True --rnd_seed=42 --filter_occluded=True --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" --map_name="Town03" > results/scrape_town03.log)

# TEST DATA
(/home/abhinav/project/CARLA_0.9.14/CarlaUE4.sh --world-port=2040 2>&1 > /dev/null &) && (sleep 15) && (python3 -u main.py scrape --outf="/home/abhinav/project/CARLA_rendering/town05" --headless=True --rnd_seed=42 --filter_occluded=True --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" --port=2040 --map_name="Town05" > results/scrape_town05.log)

sleep 64800s

tar -cvz -f /home/abhinav/project/CARLA_rendering/town03.tar.gz /home/abhinav/project/CARLA_rendering/town03
tar -cvz -f /home/abhinav/project/CARLA_rendering/town05.tar.gz /home/abhinav/project/CARLA_rendering/town05
mv /home/abhinav/project/CARLA_rendering/town03.tar.gz /media/abhinav/baap2/abhinav/datasets/viewpoint${2}${1}/
mv /home/abhinav/project/CARLA_rendering/town05.tar.gz /media/abhinav/baap2/abhinav/datasets/viewpoint${2}${1}/
