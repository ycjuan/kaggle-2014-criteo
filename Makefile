all: gbdt ffm tr.csv te.csv fc.trva.t10.txt

gbdt:
	make -C solvers/gbdt
	ln -sf solvers/gbdt/gbdt

ffm:
	make -C solvers/ffm
	ln -sf solvers/ffm/ffm

tr.csv:
	ln -s train.csv tr.csv

te.csv:
	./utils/add_dummy_label.py test.csv te.csv

fc.trva.t10.txt:
	./utils/count.py tr.csv > fc.trva.t10.txt

clean:
	rm -f gbdt ffm fc.trva.t10.txt submission.csv *.sp* te.csv tr.csv
	make -C solvers/gbdt clean
	make -C solvers/ffm clean
