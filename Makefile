
.PHONY : clean reweighting

RUNDIR=run
CONFIG=config.yaml
CONFIG2D=config2d.yaml # config file with latent space of dimension 2
CONFIGDCA=config_dca.yaml

CONDA_PYTORCH_ENV=pytorch-docker


check_env:
	# check that we have the right conda environment
	ifneq ($(CONDA_DEFAULT_ENV), $(CONDA_PYTORCH_ENV))
	$(error CONDA needs to be activated in $(CONDA_PYTORCH_ENV) environment )
	endif

clean:
	rm -f working/*.npy \
		  working/*.png \
		  working/*.pt \
		  working/*.pkl 

deepclean: clean
	rm -f output/*

reweighting: check_env
	${RUNDIR}/reweighting.sh ../${CONFIG}

runmodel: check_env
	${RUNDIR}/runmodel.sh ../${CONFIG}

plotlatent: check_env
	${RUNDIR}/plotlatent.sh ../${CONFIG2D}

rundca: check_env
	${RUNDIR}/rundca.sh "../${CONFIGDCA}"

staging.tar.gz : 
	tar --exclude='.git' \
			--exclude='.gitignore' \
			-czvf staging.tar.gz \
			./source \
			./configs\
			run/runmodel.sh\
			run/reweighting.sh

sequences.tar.gz :
	tar -czvf /squid/jlwang5/sequences.tar.gz ./sequence_sets

chtc_submit_reweighting: staging.tar.gz sequences.tar.gz
	condor_submit \
			run/chtc_reweighting.sub
submit_training: staging.tar.gz sequences.tar.gz
	condor_submit \
			run/chtc_train_model.sub
