#override FLAGS += --shell-escape
#override FLAGS += -interaction=nonstopmode 
#override FLAGS += -interaction=nonstopmode --shell-escape

MAIN_PDF=slides.pdf

all: $(MAIN_PDF)


%.pdf : %.tex
	latexmk -pdf $<

quick: 
	@latexmk -pdf -g $(MAIN_PDF:.pdf=.tex)

clean:
	@latexmk -c
	@rm -f *-figure*.* template.zip slides.bbl slides.nav slides.run.xml slides.snm

dist_clean:
	@latexmk -C
	@rm -f *-figure*.* template.zip slides.bbl slides.nav slides.run.xml slides.snm

template.zip: 
	@zip -FS -r $@ slides.tex references.bib Makefile latexmkrc tuil_theme/ figures/ -x *_orig.pdf -x *_orig.jpg -x *.svg

.PHONY: quick all clean dist_clean template.zip
