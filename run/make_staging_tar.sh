cd ../..
tar -czf VAEs.tar.gz VAEs --exclude *logs/* --exclude *sequence/* --exclude *.tar.gz
mv VAEs.tar.gz VAEs/run
