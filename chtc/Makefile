

clean:
	rm -f logs/*
	rm -f staging.tar.gz

staging:
	tar --exclude='./chtc' --exclude='.git' -czvf staging.tar.gz -C ../ ./

submit_reweighting:
	condor_submit pytorch_reweighting.sub
