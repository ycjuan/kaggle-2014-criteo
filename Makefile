all: gbdt fm tr.csv te.csv fc.trva.t10.txt

gbdt:
	make -C solvers/gbdt
	ln -sf solvers/gbdt/gbdt

fm:
	make -C solvers/fm
	ln -sf solvers/fm/fm

tr.csv:
	ln -s train.csv tr.csv

te.csv:
	./utils/add_dummy_label.py test.csv te.csv

fc.trva.t10.txt:
	./utils/count.py tr.csv > fc.trva.t10.txt

clean:
	rm -f gbdt fm fc.trva.t10.txt submission.csv *.fm* te.csv tr.csv
	make -C solvers/gbdt clean
	make -C solvers/fm clean
