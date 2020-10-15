
.PHONY: clean deepclean chtc_submit_reweighting chtc_submit_training

CONFIG := config.yaml

# clean logs and error files and code that is staged
clean:
	rm -rf logs/*
	rm -f staging.tar.gz sequences.tar.gz
	rm -f docker_stderror
	rm -f config.yaml

# clean all output files 
deepclean: clean
	rm -f *output*.tar.gz

# All the scripts in the run directory
# This should be copied over to the staging archive 
# so that they can be found in the run directory on chtc
RUNSCRIPTS := $(wildcard *.sh)

# Rule to make the staging archive which is sent to chtc jobs
# The staging archive is made in such a way that it expands out to almost the
# same layout as the github repo
# For example: the -C ../ option below changes the archive to create paths
# 				relative to the parent directory
staging.tar.gz :
	tar --exclude='.git' \
			--transform='s/$(CONFIG)/config.yaml/' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz -C ../ \
			./source \
			$(CONFIG) \
			$(addprefix run/, $(RUNSCRIPTS))

SEQUENCE_SET=$(shell ./get_config_chtc.sh ../$(CONFIG) --aligned_msa_filename)
sequences.tar.gz :
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf sequences.tar.gz -C ../ \
			sequence_sets/$(SEQUENCE_SET)

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
