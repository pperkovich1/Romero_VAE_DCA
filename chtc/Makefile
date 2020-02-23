
.PHONY: clean deepclean submit_reweighting

# clean logs and error files and code that is staged
clean:
	rm -f logs/*
	rm -f staging.tar.gz
	rm -f docker_stderror

# clean all output files 
deepclean: clean
	rm -f output*.tar.gz

# Rule to make the staging archive which is sent to chtc jobs
staging.tar.gz :
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz -C ../ \
			./source \
			./sequence_sets

# submit jobs
submit_reweighting: staging.tar.gz
	condor_submit \
			pytorch_reweighting.sub
	condor_submit \
			DISABLE_FLOCKING=1 \
			DISABLE_GPU=1 \
			pytorch_reweighting.sub

