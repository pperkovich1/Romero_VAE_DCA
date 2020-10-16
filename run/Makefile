
.PHONY: clean deepclean chtc_run

# default config file but it can be changed on the command line
CONFIG := config.yaml
# default directory that the config file is in (relative to this makefile)
# or can be an absolute path
CONFIG_DIR := ..
# can be changed on the command line to any target in the Makefile in the 
# parent directory
CHTC_MAKE_COMMAND = reweighting


CONFIG_DIR_REAL = $(realpath $(CONFIG_DIR))

# empty target
Nothing: 


# clean logs and error files and code that is staged
clean:
	rm -rf logs/*
	rm -f staging.tar.gz sequences.tar.gz
	rm -f docker_stderror

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
			Makefile \
			$(addprefix run/, $(RUNSCRIPTS)) \
			-C $(CONFIG_DIR_REAL) \
			"$(CONFIG)" \

SEQUENCE_SET := $(shell ./get_config_chtc.sh ../$(CONFIG) --aligned_msa_filename)
sequences.tar.gz :
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf sequences.tar.gz -C ../ \
			sequence_sets/"$(SEQUENCE_SET)"

# submit jobs
chtc_run: staging.tar.gz sequences.tar.gz
	# Example submit for running on CPU only
	#condor_submit \
	#		DISABLE_FLOCKING=1 \
	#		DISABLE_GPU=1 \ 
	#		chtc_run.sub
	# Run on GPU instead (This is the default)
	condor_submit CHTC_MAKE_COMMAND=$(CHTC_MAKE_COMMAND) chtc_run.sub


