
.PHONY : clean reweighting check_env

RUNDIR=run
CONFIG=config.yaml
CONFIG2D=config2d.yaml # config file with latent space of dimension 2
CONFIGDCA=config_dca.yaml

CHECK_CONDA=1
CONDA_PYTORCH_ENV=pytorch-docker

check_env:
ifeq ($(CHECK_CONDA), 1)
	@echo "Checking for Conda environment by default. Disable this with CHECK_CONDA=0"
	# check that we have the right conda environment
ifneq ($(CONDA_DEFAULT_ENV), $(CONDA_PYTORCH_ENV))
	$(error CONDA needs to be activated in $(CONDA_PYTORCH_ENV) environment )
endif
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

reweight_and_run: check_env
	${RUNDIR}/reweighting.sh ../${CONFIG}
	${RUNDIR}/runmodel.sh ../${CONFIG}

plotlatent: check_env
	${RUNDIR}/plotlatent.sh ../${CONFIG2D}

rundca: check_env
	${RUNDIR}/rundca.sh "../${CONFIGDCA}"

