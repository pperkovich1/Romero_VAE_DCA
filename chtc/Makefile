

clean:
	rm -f logs/*
	rm -f staging.tar.gz
	rm -f docker_stderror

deepclean: clean
	rm -f output.tar.gz

staging:
	tar --exclude='./chtc' \
			--exclude='.git' \
			--exclude='.gitignore' \
			--exclude='README.md' \
			-czvf staging.tar.gz -C ../ ./

submit_reweighting: 
	condor_submit pytorch_reweighting.sub
