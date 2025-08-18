angle=$1
extrinsic=$2
# angle, extrinsic, yaw, pitch, height

project_dir="/home/abhinav/project/CARLA_rendering/"
final_dir="/home/abhinav/3d_datasets/carla_abhinav/"
carla_binary="/home/abhinav/project/CARLA_0.9.14/CarlaUE4.sh"

log_file_dir="logs"
town03_name="town03"
town05_name="town05"
arxiv_suffix=".tar.gz"

town03_folder=$project_dir/${2}${1}/$town03_name
town05_folder=$project_dir/${2}${1}/$town05_name

town03_arxiv=$project_dir/$town03_name$arxiv_suffix
town05_arxiv=$project_dir/$town05_name$arxiv_suffix

output_dir=$final_dir${2}${1}

log_file_base=$log_file_dir/${2}${1}
town03_log_file=$log_file_base$town03_name
town05_log_file=$log_file_base$town05_name

#cp -r /tzofi/repos/scrape-bev-carla repo
cd $project_dir

mkdir -p $town03_folder
mkdir -p $town05_folder
mkdir -p $output_dir
mkdir -p $log_file_dir

# TEST DATA
($carla_binary -RenderOffScreen --world-port=2040 2>&1 > /dev/null &) && (sleep 15) && (python3 -u main.py scrape --outf=$town05_folder --headless=True --rnd_seed=42 --filter_occluded=True --port=2040 --map_name="Town05" --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" > $town05_log_file)

# TRAIN DATA
($carla_binary -RenderOffScreen 2>&1 > /dev/null &)                   && (sleep 15) && (python3 -u main.py scrape --outf=$town03_folder --headless=True --rnd_seed=42 --filter_occluded=True             --map_name="Town03" --cam_yaw_adjust="${3}" --cam_pitch_adjust="${4}" --cam_height_adjust="${5}" > $town03_log_file)

#tar -cvz -f $town03_arxiv $town03_folder
#tar -cvz -f $town05_arxiv $town05_folder
#mv $town03_arxiv $output_dir
#mv $town05_arxiv $output_dir
mv $town03_folder $output_dir
mv $town05_folder $output_dir
