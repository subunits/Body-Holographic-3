\
    /* holo_full.c
     * Forward hologram generation and reconstruction (C)
     * Requires: FFTW3, libpng
     *
     * Build:
     *   gcc holo_full.c -o holo_full -lfftw3 -lpng -lm
     *
     * Usage examples in README.
     */
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    #include <math.h>
    #include <png.h>
    #include <fftw3.h>
    #ifndef M_PI
    #define M_PI 3.14159265358979323846
    #endif

    static void abort_msg(const char *m) { fprintf(stderr, "%s\n", m); exit(1); }

    // --- PNG read/write helpers (8-bit grayscale) ---
    static int read_png_gray8(const char *filename, unsigned char **data, int *w, int *h) {
        FILE *fp = fopen(filename, "rb"); if (!fp) return 0;
        png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png) { fclose(fp); return 0; }
        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_read_struct(&png, NULL, NULL); fclose(fp); return 0; }
        if (setjmp(png_jmpbuf(png))) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 0; }
        png_init_io(png, fp); png_read_info(png, info);
        png_uint_32 width, height; int bit_depth, color_type;
        png_get_IHDR(png, info, &width, &height, &bit_depth, &color_type, NULL, NULL, NULL);
        // convert to 8-bit gray
        if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
        if (color_type == PNG_COLOR_TYPE_RGB || color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_rgb_to_gray_fixed(png, 1, -1, -1);
        if (color_type & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(png);
        if (bit_depth == 16) png_set_strip_16(png);
        png_read_update_info(png, info);
        *w = (int)width; *h = (int)height;
        *data = (unsigned char*) malloc((*w) * (*h));
        if (!*data) { png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 0; }
        png_bytep *rows = (png_bytep*) malloc(sizeof(png_bytep) * (*h));
        if (!rows) { free(*data); png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 0; }
        for (int y=0;y<*h;y++) rows[y] = (*data) + y*(*w);
        png_read_image(png, rows); png_read_end(png, NULL);
        free(rows); png_destroy_read_struct(&png, &info, NULL); fclose(fp); return 1;
    }

    static int save_png_gray8(const char *filename, unsigned char *data, int w, int h) {
        FILE *fp = fopen(filename, "wb"); if (!fp) { perror("fopen"); return 0; }
        png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        if (!png) { fclose(fp); return 0; }
        png_infop info = png_create_info_struct(png);
        if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return 0; }
        if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return 0; }
        png_init_io(png, fp);
        png_set_IHDR(png, info, w, h, 8, PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_write_info(png, info);
        png_bytep *rows = (png_bytep*) malloc(sizeof(png_bytep) * h);
        if (!rows) { png_destroy_write_struct(&png, &info); fclose(fp); return 0; }
        for (int y=0;y<h;y++) rows[y] = data + y*w;
        png_write_image(png, rows); png_write_end(png, NULL);
        free(rows); png_destroy_write_struct(&png, &info); fclose(fp); return 1;
    }

    static int save_double_as_png(const char *fname, double *buf, int w, int h) {
        unsigned char *out = (unsigned char*) malloc(w*h);
        if (!out) return 0;
        double mn = buf[0], mx = buf[0];
        for (int i=0;i<w*h;i++) { if (buf[i]<mn) mn = buf[i]; if (buf[i]>mx) mx = buf[i]; }
        double scale = (mx>mn)? 255.0/(mx-mn) : 1.0;
        for (int i=0;i<w*h;i++) {
            double v = (buf[i]-mn)*scale;
            if (v<0) v=0; if (v>255) v=255;
            out[i] = (unsigned char) lrint(v);
        }
        int ok = save_png_gray8(fname, out, w, h);
        free(out); return ok;
    }

    // Simple object generators
    static double obj_gauss3(double x, double y, int NX, int NY) {
        double xx = x - NX/2.0, yy = y - NY/2.0;
        double r1 = exp(-((xx+30)*(xx+30) + (yy+20)*(yy+20)) / 200.0);
        double r2 = exp(-((xx-40)*(xx-40) + (yy-25)*(yy-25)) / 300.0);
        double r3 = exp(-(xx*xx + yy*yy) / 500.0);
        return r1 + r2 + r3;
    }
    static double obj_circle(double x, double y, int NX, int NY, double R) {
        double dx = x - NX/2.0, dy = y - NY/2.0;
        return (dx*dx + dy*dy <= R*R) ? 1.0 : 0.0;
    }

    int main(int argc, char **argv) {
        if (argc < 2) { printf("Usage: %s forward <options> | recon <options>\\n", argv[0]); return 1; }
        const char *cmd = argv[1];
        // defaults
        int NX = 512, NY = 512;
        double kx = 0.05, ky = 0.03;
        int mode = 0; // 0 intensity,1 amp,2 phase
        const char *obj = "gauss3";
        double circleR = 50.0;
        const char *in_png = NULL;
        const char *out_png = "hologram.png";
        double wavelength = 532e-9;
        double dx = 6.5e-6;
        double z = 0.08;
        int prop_method = 0; // 0 angular, 1 fresnel
        // parse simple args
        for (int i=2;i<argc;i++) {
            if (!strcmp(argv[i],"-nx") && i+1<argc) NX = atoi(argv[++i]);
            else if (!strcmp(argv[i],"-ny") && i+1<argc) NY = atoi(argv[++i]);
            else if (!strcmp(argv[i],"-kx") && i+1<argc) kx = atof(argv[++i]);
            else if (!strcmp(argv[i],"-ky") && i+1<argc) ky = atof(argv[++i]);
            else if (!strcmp(argv[i],"-mode") && i+1<argc) mode = atoi(argv[++i]);
            else if (!strcmp(argv[i],"-obj") && i+1<argc) obj = argv[++i];
            else if (!strcmp(argv[i],"-r") && i+1<argc) circleR = atof(argv[++i]);
            else if (!strcmp(argv[i],"-in") && i+1<argc) in_png = argv[++i];
            else if (!strcmp(argv[i],"-out") && i+1<argc) out_png = argv[++i];
            else if (!strcmp(argv[i],"-z") && i+1<argc) z = atof(argv[++i]);
            else if (!strcmp(argv[i],"-prop") && i+1<argc) { if (!strcmp(argv[++i],"fresnel")) prop_method = 1; else prop_method = 0; }
        }

        if (!strcmp(cmd,"forward")) {
            // allocate
            fftw_complex *U = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);
            fftw_complex *Ufft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * NX * NY);
            double *tmp = (double*) malloc(sizeof(double) * NX * NY);
            if (!U || !Ufft || !tmp) abort_msg("alloc fail");
            // build object
            if (!strcmp(obj,"png")) {
                if (!in_png) abort_msg("in_png required for obj=png");
                unsigned char *img; int iw, ih;
                if (!read_png_gray8(in_png, &img, &iw, &ih)) abort_msg("failed read input png");
                // simple nearest neighbor resize
                for (int y=0;y<NY;y++) for (int x=0;x<NX;x++) {
                    int sx = (int)( (double)x * iw / NX ); if (sx>=iw) sx = iw-1;
                    int sy = (int)( (double)y * ih / NY ); if (sy>=ih) sy = ih-1;
                    double amp = img[sy*iw + sx] / 255.0;
                    U[y*NX + x][0] = amp; U[y*NX + x][1] = 0.0;
                }
                free(img);
            } else if (!strcmp(obj,"circle")) {
                for (int y=0;y<NY;y++) for (int x=0;x<NX;x++) { U[y*NX + x][0] = obj_circle(x,y,NX,NY,circleR); U[y*NX + x][1]=0.0; }
            } else {
                for (int y=0;y<NY;y++) for (int x=0;x<NX;x++) { U[y*NX + x][0] = obj_gauss3(x,y,NX,NY); U[y*NX + x][1]=0.0; }
            }
            // FFT
            fftw_plan pf = fftw_plan_dft_2d(NY,NX,U,Ufft,FFTW_FORWARD,FFTW_ESTIMATE);
            fftw_execute(pf); fftw_destroy_plan(pf);
            // form hologram with reference tilt kx,ky (radians per pixel)
            double maxv = 0.0;
            for (int y=0;y<NY;y++) for (int x=0;x<NX;x++) {
                int idx = y*NX + x;
                double Ore = Ufft[idx][0], Oim = Ufft[idx][1];
                double phase = kx * x + ky * y;
                double Rre = cos(phase), Rim = sin(phase);
                double Fre = Ore + Rre, Fim = Oim + Rim;
                double amp = sqrt(Fre*Fre + Fim*Fim);
                if (mode==0) tmp[idx] = amp*amp;
                else if (mode==1) tmp[idx] = amp;
                else tmp[idx] = atan2(Fim, Fre); // raw phase [-pi,pi]
                if (mode!=2 && tmp[idx] > maxv) maxv = tmp[idx];
            }
            // normalize & save
            if (mode==2) { for (int i=0;i<NX*NY;i++) tmp[i] = (tmp[i] + M_PI) / (2.0*M_PI); }
            else if (maxv>0.0) for (int i=0;i<NX*NY;i++) tmp[i] /= maxv;
            if (!save_double_as_png(out_png, tmp, NX, NY)) abort_msg("save png failed");
            printf("Wrote %s\n", out_png);
            fftw_free(U); fftw_free(Ufft); free(tmp);
            return 0;
        } else if (!strcmp(cmd,"recon")) {
            // Reconstruction from hologram PNG (reads intensity hologram)
            if (!in_png) abort_msg("specify -in hologram.png");
            unsigned char *himg; int iw, ih;
            if (!read_png_gray8(in_png, &himg, &iw, &ih)) abort_msg("failed read hologram PNG");
            // Convert to double amplitude estimate
            double *I = (double*) malloc(sizeof(double)*iw*ih);
            for (int i=0;i<iw*ih;i++) I[i] = himg[i]/255.0;
            free(himg);
            // Use sqrt(I) as amplitude (simple)
            double *A = (double*) malloc(sizeof(double)*iw*ih);
            for (int i=0;i<iw*ih;i++) A[i] = sqrt(I[i]);
            // Now perform FFT, multiply by transfer function and inverse FFT
            // For simplicity, do angular in frequency domain using FFTW 2D
            fftw_complex *U = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*iw*ih);
            fftw_complex *F = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*iw*ih);
            for (int i=0;i<iw*ih;i++) { U[i][0] = A[i]; U[i][1] = 0.0; }
            fftw_plan pf = fftw_plan_dft_2d(ih,iw,U,F,FFTW_FORWARD,FFTW_ESTIMATE);
            fftw_execute(pf); fftw_destroy_plan(pf);
            // apply propagation kernel (angular spectrum) for -z
            double k = 2*M_PI / wavelength;
            for (int y=0;y<ih;y++) {
                double fy = (y < ih/2) ? (double)y/ (ih*dx) : (double)(y-ih)/(ih*dx);
                for (int x=0;x<iw;x++) {
                    double fx = (x < iw/2) ? (double)x/ (iw*dx) : (double)(x-iw)/(iw*dx);
                    double fsq = (wavelength*fx)*(wavelength*fx) + (wavelength*fy)*(wavelength*fy);
                    double root = 0.0;
                    if (fsq < 1.0) root = sqrt(1.0 - fsq);
                    double phase = k * (-z) * root;
                    double cre = cos(phase), cim = sin(phase);
                    int idx = y*iw + x;
                    double re = F[idx][0], im = F[idx][1];
                    double rre = re*cre - im*cim;
                    double rim = re*cim + im*cre;
                    F[idx][0] = rre; F[idx][1] = rim;
                }
            }
            // inverse FFT
            fftw_plan pi = fftw_plan_dft_2d(ih,iw,F,U,FFTW_BACKWARD,FFTW_ESTIMATE);
            fftw_execute(pi); fftw_destroy_plan(pi);
            // extract amplitude and phase
            double *amp = (double*) malloc(sizeof(double)*iw*ih);
            double *phs = (double*) malloc(sizeof(double)*iw*ih);
            for (int i=0;i<iw*ih;i++) {
                double re = U[i][0]/(iw*ih), im = U[i][1]/(iw*ih);
                amp[i] = sqrt(re*re + im*im);
                phs[i] = atan2(im, re);
            }
            // save outputs
            save_double_as_png("recon_amp.png", amp, iw, ih);
            // normalize phase to 0..1
            for (int i=0;i<iw*ih;i++) phs[i] = (phs[i] + M_PI) / (2.0*M_PI);
            save_double_as_png("recon_phase.png", phs, iw, ih);
            printf("Wrote recon_amp.png and recon_phase.png\n");
            // free
            free(I); free(A); free(amp); free(phs);
            fftw_free(U); fftw_free(F);
            return 0;
        } else {
            printf("Unknown cmd. Use forward or recon.\n"); return 1;
        }
    }
