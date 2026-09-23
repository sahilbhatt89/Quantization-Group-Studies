SVG_FILES = $(shell find . -type f -name '*.svg')
PDF_FILES = $(SVG_FILES:.svg=.pdf)

EPS_FILES = $(shell find . -type f -name '*.eps')
EPSPDF_FILES = $(EPS_FILES:.eps=.pdf)

PNG_FILES = $(shell find . -type f -name '*.png')
PNGPDF_FILES = $(PNG_FILES:.png=.pdf)

DOT_FILES = $(shell find . -type f -name '*.dot')
DOTPDF_FILES = $(DOT_FILES:.dot=.pdf)


all: $(PDF_FILES) $(EPSPDF_FILES) $(PNGPDF_FILES) $(DOTPDF_FILES)

rebuilt: clean all

%.pdf: %.eps
	@echo -ne "epstopdf $< \n"; \
	epstopdf $<


%.pdf: %.svg
	@echo -ne "inkscape -z -T -A $@ $< \n"; \
	inkscape -z -T -A $@ $<

%.pdf: %.png
	@echo -ne "convert $< $@ \n"; \
	convert $< $@

%.pdf: %.dot
	@echo -ne "cat $< | dot -Tpdf > $@ \n";\
	cat $< | dot -Tpdf > $@

clean :
	rm -f $(PDF_FILES) $(EPSPDF_FILES) $(PNGPDF_FILES) $(DOTPDF_FILES)

gitignore:
	@echo $(PDF_FILES) $(EPSPDF_FILES) $(PNGPDF_FILES) $(DOTPDF_FILES) | sed "s/ /\n/g" | sed "s/.\///"  > .gitignore

#@echo $(PDF_FILES) $(EPSPDF_FILES) $(PNGPDF_FILES) $(DOTPDF_FILES) | sed "s/.\//\n/g" | tr -d ' ' > .gitignore
