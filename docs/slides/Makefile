# Build the verifier-anchored RL talk decks in two forms each:
#   <deck>.pdf            clean deck for the projector (audience)
#   <deck>-presenter.pdf  slide + speaker notes side-by-side (for pympress/pdfpc)
# Decks:
#   2026-06-23-verifier-anchored-rl-talk        full version (~14 min)
#   2026-06-23-verifier-anchored-rl-talk-10min  tight 10-minute cut
# Targets: make audience | make presenter | make all | make clean
# A presenter viewer (pympress is installed; pdfpc equivalent) shows the slide to
# the audience display and the notes (+ next slide / timer) to the presenter
# display from the -presenter PDF; the plain PDF is the no-notes fallback.
DECKS = 2026-06-23-verifier-anchored-rl-talk 2026-06-23-verifier-anchored-rl-talk-10min
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all audience presenter clean
all: audience presenter

audience: $(addsuffix .pdf,$(DECKS))

%.pdf: %.tex
	$(LATEXMK) $<

presenter:
	@for d in $(DECKS); do \
	  echo "=== presenter: $$d ==="; \
	  $(LATEXMK) -usepretex='\def\shownotes{}' -jobname=$$d-presenter $$d.tex; \
	done

clean:
	@for d in $(DECKS); do latexmk -C $$d.tex; rm -f $$d-presenter.*; done
	rm -f *.snm *.nav *.vrb *.log
