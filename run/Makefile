
.PHONY: clean deepclean chtc_submit_reweighting chtc_submit_training

# default config file but it can be changed on the command line
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

# All the scripts in the run directory are in this variable
# This should be copied over to the staging archive 
# so that they can be found in the run directory on a chtc node
RUNSCRIPTS := $(wildcard *.sh)

# Rule to make the staging archive which is sent to chtc jobs
# The staging archive is made in such a way that it expands out to almost the
# same layout as the github repo
# For example: the -C ../ option below changes the archive to create paths
# 				relative to the parent directory
# We rename the $(CONFIG) file to be config.yaml
staging.tar.gz :
	tar --exclude='.git' \
			--transform='s/$(CONFIG)/config.yaml/' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz -C ../ \
			./source \
			"$(CONFIG)" \
			Makefile \
			$(addprefix run/, $(RUNSCRIPTS))

SEQUENCE_SET := $(shell ./get_config_chtc.sh ../$(CONFIG) --aligned_msa_filename)
sequences.tar.gz :
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf sequences.tar.gz -C ../ \
			sequence_sets/"$(SEQUENCE_SET)"

# submit jobs
chtc_submit_reweighting: staging.tar.gz sequences.tar.gz
	# Example submit for running on CPU only
	#condor_submit \
	#		DISABLE_FLOCKING=1 \
	#		DISABLE_GPU=1 \ 
	#		chtc_reweighting.sub
	# Run on GPU instead (This is the default)
	condor_submit chtc_reweighting.sub


