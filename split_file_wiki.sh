for i in $(seq 1 2205 2310098)
do
	sed -n -e $i,$((i+735))p wiki_standarized_l > segment_$i_$((i+735))
done