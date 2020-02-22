

clean:
	rm -f logs/*
	rm -f staging.tar.gz
	rm -f docker_stderror

deepclean: clean
	rm -f output.tar.gz

staging:
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz -C ../ \
			./source \
			./sequence_sets

submit_reweighting: 
	condor_submit pytorch_reweighting.sub
	condor_submit CPU_ONLY=1 pytorch_reweighting.sub

