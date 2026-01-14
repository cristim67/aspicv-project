# Scientific Report

This directory contains the LaTeX source code for the scientific paper associated with this project.

## 📄 Contents
- `paper.tex`: The main LaTeX source file tailored for the IEEE format.

## 🔨 How to Build

To compile the PDF report, you need a LaTeX distribution (like TeX Live) installed.

### Using PDFLaTeX

Run the following command in your terminal:

```bash
pdflatex paper.tex
```

This will generate `paper.pdf` which contains the formatted report.

### Clean Build Artifacts

To remove intermediate files generated during compilation:

```bash
rm *.aux *.log *.out *.toc
```
