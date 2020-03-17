
.PHONY : clean reweighting

RUNDIR=run
CONFIG=config.yaml

CONDA_PYTORCH_ENV=pytorch-docker

# check that we have the right conda environment
ifneq ($(CONDA_DEFAULT_ENV), $(CONDA_PYTORCH_ENV))
$(error CONDA needs to be activated in $(CONDA_PYTORCH_ENV) environment )
endif

clean:
	rm -f working/*.npy

deepclean: clean
	rm -f output/*

reweighting:
	${RUNDIR}/reweighting.sh ../${CONFIG}
