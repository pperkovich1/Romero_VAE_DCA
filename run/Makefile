
.PHONY: clean deepclean chtc_submit_reweighting chtc_submit_training

CONFIG := config.yaml

# clean logs and error files and code that is staged
clean:
	rm -rf logs/*
	rm -f staging.tar.gz
	rm -f docker_stderror
	rm -f config.yaml

# clean all output files 
deepclean: clean
	rm -f *output*.tar.gz

# Rule to make the staging archive which is sent to chtc jobs
staging.tar.gz :
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz -C ../ \
			./source \
			./configs\
			run/runmodel.sh\
			run/reweighting.sh
			# ./sequence_sets
			# config.yaml

sequences.tar.gz :
	tar -czvf /squid/jlwang5/sequences.tar.gz ../sequence_sets

# submit jobs
chtc_submit_reweighting: staging.tar.gz sequences.tar.gz
	condor_submit \
			chtc_reweighting.sub
	#condor_submit \
	#		DISABLE_FLOCKING=1 \
	#		DISABLE_GPU=1 \ #		chtc_reweighting.sub
submit_training: staging.tar.gz sequences.tar.gz
	condor_submit \
			chtc_train_model.sub

move_output:
	rm -f docker_stderror
	rm -f config.yaml
	mv *.err logs
	mv *.out logs
	mv *.log logs
	mv *output* logs
