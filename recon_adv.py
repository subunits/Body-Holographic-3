\
    # recon_adv.py -- Reconstruction with phase unwrapping and simple twin-image suppression
    import numpy as np
    import imageio, sys, os
    import matplotlib.pyplot as plt
    from numpy.fft import fft2, ifft2, fftshift, ifftshift

    def read_img(fname):
        im = imageio.v2.imread(fname)
        if im.ndim==3:
            im = np.dot(im[...,:3], [0.2989,0.5870,0.1140])
        return im.astype(np.float64)/255.0

    def angular_spectrum_prop(U0, wavelength, z, dx):
        Ny, Nx = U0.shape
        k = 2*np.pi/wavelength
        fx = np.fft.fftfreq(Nx, dx)
        fy = np.fft.fftfreq(Ny, dx)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(1j * k * z * np.sqrt(np.maximum(0, 1 - (wavelength*FX)**2 - (wavelength*FY)**2)))
        U1 = ifft2( fft2(U0) * H )
        return U1

    def fresnel_prop(U0, wavelength, z, dx):
        Ny, Nx = U0.shape
        fx = np.fft.fftfreq(Nx, dx)
        fy = np.fft.fftfreq(Ny, dx)
        FX, FY = np.meshgrid(fx, fy)
        H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))
        U1 = ifft2( fft2(U0) * H ) * np.exp(1j*2*np.pi*z/wavelength)
        return U1

    def unwrap_phase_2d(p):
        # simple unwrap: sequential 1D unwrap along rows then columns
        p1 = np.unwrap(p, axis=1)
        p2 = np.unwrap(p1, axis=0)
        return p2

    def gerchberg_saxton_twin_suppression(holo, ref_phase, iterations=50, wavelength=532e-9, dx=6.5e-6, z=0.08):
        # holo: intensity hologram normalized 0..1
        Ny, Nx = holo.shape
        # initial estimate: amplitude = sqrt(I), random phase
        amp = np.sqrt(holo)
        rng = np.random.default_rng(0)
        phi = rng.uniform(-np.pi, np.pi, size=holo.shape)
        field = amp * np.exp(1j*phi)
        for it in range(iterations):
            # propagate to object plane (-z)
            Uobj = angular_spectrum_prop(field, wavelength, -z, dx)
            # apply object constraint: real, positive amplitude (support assumed centered)
            Uobj_amp = np.abs(Uobj)
            Uobj_phase = np.angle(Uobj)
            # simple support: threshold amplitude
            mask = Uobj_amp > 0.1*np.max(Uobj_amp)
            Uobj_new = np.where(mask, Uobj, 0)
            # propagate forward to sensor
            Usens = angular_spectrum_prop(Uobj_new, wavelength, z, dx)
            # enforce measured amplitude (sqrt(I)) but keep phase from Usens
            field = np.sqrt(holo) * np.exp(1j * np.angle(Usens))
        # final object estimate
        Uobj_final = angular_spectrum_prop(field, wavelength, -z, dx)
        return Uobj_final

    def reconstruct(holo_path, method='angular', wavelength=532e-9, dx=6.5e-6, z=0.08, do_unwrap=True, do_iterative=False):
        holo = read_img(holo_path)
        H = holo.copy()
        # estimate amplitude
        U_sensor = np.sqrt(H)
        if do_iterative:
            Uobj = gerchberg_saxton_twin_suppression(H, None, iterations=60, wavelength=wavelength, dx=dx, z=z)
        else:
            if method=='angular':
                Uobj = angular_spectrum_prop(U_sensor, wavelength, -z, dx)
            else:
                Uobj = fresnel_prop(U_sensor, wavelength, -z, dx)
        amp = np.abs(Uobj)
        phs = np.angle(Uobj)
        if do_unwrap:
            phs = unwrap_phase_2d(phs)
        return amp, phs

    if __name__=='__main__':
        if len(sys.argv) < 2:
            print("Usage: recon_adv.py hologram.png [z_m] [method=angular|fresnel] [iterative(0/1)]")
            sys.exit(1)
        holo = sys.argv[1]
        z = float(sys.argv[2]) if len(sys.argv)>2 else 0.08
        method = sys.argv[3] if len(sys.argv)>3 else 'angular'
        iterative = bool(int(sys.argv[4])) if len(sys.argv)>4 else False
        amp, phs = reconstruct(holo, method=method, z=z, do_unwrap=True, do_iterative=iterative)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1); plt.imshow(amp, cmap='gray'); plt.title('Amplitude'); plt.colorbar()
        plt.subplot(1,2,2); plt.imshow(phs, cmap='twilight'); plt.title('Phase (unwrapped)'); plt.colorbar()
        plt.show()
