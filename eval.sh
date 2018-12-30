echo $*

function get_cls() {
			python -u ../../AE/evaluation.py \
			$1 \
			--load-ckpt $2 \
			--config $3
}

mode=$1
load_path=$2
cfg=$3
start_epoch=$4

step=5000
end_epoch=1000000
#end_epoch=600
timer1=$(date +"%Y%m%d_%H%M%S")

while test $start_epoch -le $end_epoch; do
	echo ${load_path}${start_epoch}.pth.tar 
  if [  -f ${load_path}${start_epoch}.pth.tar ]; then
    get_cls ${mode} ${load_path}${start_epoch}.pth.tar ${cfg} 
    ((start_epoch+=step))
  else
   echo "sleep"
   sleep 1m 
  fi
done

