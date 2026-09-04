
# number of waveforms in a file
nwf=1000
#nwf=400
#nwf=40
# number of total files
nfile=1
#nfile=1
echo "execute 'python run_kid.py ${nwf}' with ${nfile} times"

dirname0='data'
dt=$(date +%m%d_%H%M%S)
dirname=${dirname0}_${dt}
mkdir $dirname

# save snapshots
cp -p plot_kid.py $dirname/plot.py
cp -p kid.py $dirname/
cp -p run_kid.sh $dirname/
cp -p check_wf.py $dirname/
cp -p hist.py $dirname/
cp -p condition $dirname/

ii=-1
while [ $((ii+=1)) -le $((nfile-1)) ]
do
	echo -n $ii
	python.exe kid.py ${nwf}
	mv wf_*.npz $dirname/
done

mv $dirname alldata/
