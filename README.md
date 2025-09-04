~~~
FFT Hologram — Complete Implementation (Roadmap Executed)
========================================================

This package includes:
- holo_full.c  : C program (forward hologram generation; reconstruction from hologram PNG)
- recon_adv.py : Python reconstruction with phase unwrapping and iterative twin-image suppression
- Makefile     : build helper
- README       : this file

Build dependencies:
- Linux (Debian/Ubuntu): sudo apt-get install build-essential libfftw3-dev libpng-dev python3-pip
- Python: pip3 install numpy imageio matplotlib

Build:
  make

Examples:
1) Forward hologram (C):
  ./holo_full forward -nx 512 -ny 512 -mode 0 -kx 0.05 -ky 0.03 -obj gauss3 -out holo.png

2) Reconstruct (C; simpler angular method):
  ./holo_full recon -in holo.png -z 0.08

3) Advanced reconstruction (Python, with unwrapping and iterative twin suppression):
  python3 recon_adv.py holo.png 0.08 angular 1

Notes:
- The C recon reads the hologram PNG, assumes intensity hologram, estimates amplitude = sqrt(I), and back-propagates using angular spectrum.
- The Python reconstruction offers more features (unwrapping, iterative GS suppression) and is recommended for experimentation.
~~~
