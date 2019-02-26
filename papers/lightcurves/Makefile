LATEX = pdflatex
MAIN  = main

all: $(MAIN).pdf

$(MAIN).pdf: 	$(MAIN).tex  bibliography.bib
	$(LATEX) $(MAIN).tex 
	bibtex $(MAIN)
	$(LATEX) $(MAIN).tex 
	$(LATEX) $(MAIN).tex 

diff:	$(MAIN)_diff.pdf

$(MAIN)_diff.pdf: $(MAIN).tex
	latexdiff $(MAIN)_old.tex $(MAIN).tex > $(MAIN)_diff.tex
	$(LATEX) $(MAIN)_diff.tex 
	bibtex $(MAIN)_diff
	$(LATEX) $(MAIN)_diff.tex 
	$(LATEX) $(MAIN)_diff.tex 

tar : 
	tar -czvf $(MAIN).tar.gz $(MAIN).tex Dwarfs.bib *.pdf


clean:
	rm -f $(MAIN).pdf $(MAIN).out $(MAIN).bbl $(MAIN).aux $(MAIN)_diff.pdf $(MAIN).blg $(MAIN).log 

view: 
	open $(MAIN).pdf &

